use num_complex::Complex32;
use seify_hackrfone::{Config, HackRf};
use std::sync::{
    Arc,
    atomic::AtomicBool,
    mpsc::{Receiver, Sender},
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

use crate::{RADIO_SAMPLE_RATE_HZ, SHIFT_HZ, dsp::fft::Spectrum};

const FFT_SIZE: usize = 1024; // FFT bins == waterfall texture width
const HISTORY: u32 = 512; // texture height: rows of history kept
const AVG_FFTS: usize = 64; // FFTs averaged into one row (~36 rows/s)
const DB_MIN: f32 = -80.0;
const DB_MAX: f32 = 0.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    write_row: u32,
    history: u32,
    db_min: f32,
    db_max: f32,
}

struct App {
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    rx: Receiver<Vec<f32>>,
    stop: Arc<AtomicBool>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // only build the window one time
        }
        let attrs = Window::default_attributes().with_title("waterfall");
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let gfx = pollster::block_on(Gfx::new(window.clone())).expect("wgpu init");
        window.request_redraw(); // first frame
        self.window = Some(window);
        self.gfx = Some(gfx);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(window), Some(gfx)) = (self.window.as_ref(), self.gfx.as_mut()) else {
            return;
        };
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
                event_loop.exit();
            }
            // WindowEvent::Resized(s) => gfx.resize(s.width, s.height),
            WindowEvent::RedrawRequested => {
                while let Ok(row) = self.rx.try_recv() {
                    gfx.push_row(&row);
                }
                if let Err(e) = gfx.render() {
                    eprintln!("render error: {e:#}");
                }
                window.request_redraw();
            }
            _ => {}
        }
    }
}

pub fn run(freq_hz: u64) -> anyhow::Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
    spawn_radio(freq_hz, tx, stop.clone());

    let event_loop = EventLoop::new()?;
    let mut app = App {
        window: None,
        gfx: None,
        rx,
        stop,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    waterfall: wgpu::Texture,
    params_buf: wgpu::Buffer,
    write_row: u32,
}
impl Gfx {
    async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        // cloning the 'static arc lets the surface outlive this function
        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("vidman device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let waterfall = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("waterfall"),
            size: wgpu::Extent3d {
                width: FFT_SIZE as u32,
                height: HISTORY,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let waterfall_view = waterfall.create_view(&wgpu::TextureViewDescriptor::default());

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("waterfall bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&waterfall_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("waterfall.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("waterfall.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("waterfall layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("waterfall pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            bind_group,
            waterfall,
            params_buf,
            write_row: 0,
        })
    }

    fn push_row(&mut self, row: &[f32]) {
        debug_assert_eq!(row.len(), FFT_SIZE);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.waterfall,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: self.write_row,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(row),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((FFT_SIZE * std::mem::size_of::<f32>()) as u32),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: FFT_SIZE as u32,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.write_row = (self.write_row + 1) % HISTORY;
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w > 0 && h > 0 {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn render(&mut self) -> anyhow::Result<()> {
        self.queue.write_buffer(
            &self.params_buf,
            0,
            bytemuck::bytes_of(&Params {
                write_row: self.write_row,
                history: HISTORY,
                db_min: DB_MIN,
                db_max: DB_MAX,
            }),
        );

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            other => anyhow::bail!("surface acquire failed: {other:?}"),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("waterfall render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: Default::default(),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
        frame.present();
        Ok(())
    }
}

fn spawn_radio(freq_hz: u64, tx: Sender<Vec<f32>>, stop: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        let hackrf = HackRf::open_first().expect("open hackrf");
        hackrf
            .start_rx(&Config {
                txvga_db: 0,
                vga_db: 16,
                lna_db: 16,
                amp_enable: false,
                antenna_enable: false,
                frequency_hz: freq_hz + SHIFT_HZ,
                sample_rate_hz: RADIO_SAMPLE_RATE_HZ,
                sample_rate_div: 1,
            })
            .expect("start_rx");

        let mut spec = Spectrum::new(FFT_SIZE);
        let mut buf = vec![0u8; 262_144];
        let mut chunk = vec![Complex32::new(0.0, 0.0); FFT_SIZE];
        let mut pushed = 0usize;

        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
            let n = match hackrf.read(&mut buf) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("hackrf read error: {e:#}");
                    continue;
                }
            };

            let samples = n / 2;
            let mut s = 0;
            while s + FFT_SIZE <= samples {
                for k in 0..FFT_SIZE {
                    let re = (buf[2 * (s + k)] as i8) as f32 / 128.0;
                    let im = (buf[2 * (s + k) + 1] as i8) as f32 / 128.0;
                    chunk[k] = Complex32::new(re, im);
                }
                spec.push(&chunk);
                pushed += 1;
                s += FFT_SIZE;

                if pushed == AVG_FFTS {
                    // take(): db spectrum averaged over avg_ffts, fftshifted.
                    if tx.send(spec.take()).is_err() {
                        let _ = hackrf.stop(); // receiver gone — window closed
                        return;
                    }
                    pushed = 0;
                }
            }
        }
        let _ = hackrf.stop();
    });
}
