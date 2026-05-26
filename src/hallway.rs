use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, Ordering},
    mpsc::{self, Receiver, Sender, SyncSender},
};

use glam::{Mat4, Vec3};
use num_complex::Complex32;
use rtrb::{Producer, RingBuffer};
use seify_hackrfone::{Config, HackRf};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::{
    ControlMsg, RADIO_SAMPLE_RATE_HZ, SHIFT_HZ, audio,
    dsp::{fft::Spectrum, pipeline::Pipeline},
    spawn_stdin_reader,
};

const FFT_SIZE: usize = 1024;
const HISTORY: u32 = 512;
const AVG_FFTS: usize = 64;
const DB_MIN: f32 = -65.0;
const DB_MAX: f32 = -15.0;

const HZ_PER_METER: f32 = 50_000.0;
const HALLWAY_WIDTH: f32 = 4.0;
const HALLWAY_HEIGHT: f32 = 3.0;
const HEAD_HEIGHT: f32 = 1.7;
const WALK_SPEED: f32 = 3.0; // m/s
const TURN_SPEED: f32 = 1.8; // rad/s

// world coordinate system: x in meters, origin pinned to the initial HackRF LO
// frequency. freq_at_world(x) = initial_lo_hz + x * HZ_PER_METER.
const WALL_HALF_LEN_M: f32 = 200.0; // walls extend ±200 m world (~±10 MHz of slop)
const BAND_HALF_WIDTH_M: f32 = (RADIO_SAMPLE_RATE_HZ as f32) / 2.0 / HZ_PER_METER;
// 1 m ≈ 50 kHz — fine enough to feel like the radio is following the camera,
// coarse enough to not retune every block.
const RETUNE_THRESHOLD_M: f32 = 1.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CamUniform {
    view_proj: [[f32; 4]; 4],     // 64 bytes
    write_row: u32,               // 68
    history: u32,                 // 72
    db_min: f32,                  // 76
    db_max: f32,                  // 80
    band_half_width_m: f32,       // 84
    _pad0: f32,                   // 88
    _pad1: f32,                   // 92
    _pad2: f32,                   // 96
}

/// One row of FFT data plus the LO position when it was captured.
struct SpecRow {
    samples: Vec<f32>,
    lo_world_x: f32,
}

fn store_f32(a: &AtomicU32, v: f32) {
    a.store(v.to_bits(), Ordering::Relaxed);
}
fn load_f32(a: &AtomicU32) -> f32 {
    f32::from_bits(a.load(Ordering::Relaxed))
}

struct Camera {
    pos: Vec3,
    yaw: f32,
    pitch: f32,
    fov_y: f32,
    aspect: f32,
}

impl Camera {
    fn view_proj(&self) -> Mat4 {
        let forward = Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        );
        let view = Mat4::look_to_rh(self.pos, forward, Vec3::Y);
        let proj = Mat4::perspective_rh(self.fov_y, self.aspect, 0.1, 1000.0);
        proj * view
    }
}

#[derive(Default)]
struct Input {
    forward: bool,
    back: bool,
    yaw_left: bool,
    yaw_right: bool,
}

impl Input {
    fn handle(&mut self, ev: &WindowEvent) {
        if let WindowEvent::KeyboardInput {
            event:
                KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state,
                    ..
                },
            ..
        } = ev
        {
            let down = *state == ElementState::Pressed;
            match code {
                KeyCode::KeyW | KeyCode::ArrowUp => self.forward = down,
                KeyCode::KeyS | KeyCode::ArrowDown => self.back = down,
                KeyCode::KeyA | KeyCode::ArrowLeft => self.yaw_left = down,
                KeyCode::KeyD | KeyCode::ArrowRight => self.yaw_right = down,
                _ => {}
            }
        }
    }

    fn apply(&self, cam: &mut Camera, dt: f32) {
        if self.forward {
            cam.pos.x += WALK_SPEED * dt;
        }
        if self.back {
            cam.pos.x -= WALK_SPEED * dt;
        }
        if self.yaw_left {
            cam.yaw -= TURN_SPEED * dt;
        }
        if self.yaw_right {
            cam.yaw += TURN_SPEED * dt;
        }
    }
}

struct App {
    window: Option<Arc<Window>>,
    gfx: Option<Gfx>,
    rx: Receiver<SpecRow>,
    ctrl_rx: Receiver<ControlMsg>,
    rebase_tx: Sender<u64>,
    stop: Arc<AtomicBool>,
    camera: Camera,
    input: Input,
    last_frame: std::time::Instant,
    shared_camera_x: Arc<AtomicU32>,
    shared_lo_world_x: Arc<AtomicU32>,
    initial_freq_hz: u64,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_title("vidman hallway");
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let gfx = pollster::block_on(Gfx::new(window.clone())).expect("wgpu init");
        let size = window.inner_size();
        self.camera.aspect = size.width as f32 / size.height.max(1) as f32;
        window.request_redraw();
        self.window = Some(window);
        self.gfx = Some(gfx);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (Some(window), Some(gfx)) = (self.window.as_ref(), self.gfx.as_mut()) else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => {
                self.stop.store(true, Ordering::Relaxed);
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.stop.store(true, Ordering::Relaxed);
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { .. } => {
                self.input.handle(&event);
            }
            WindowEvent::Resized(s) => {
                gfx.resize(s.width, s.height);
                self.camera.aspect = s.width as f32 / s.height.max(1) as f32;
            }
            WindowEvent::RedrawRequested => {
                let now = std::time::Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;
                self.input.apply(&mut self.camera, dt);

                // drain stdin warp/retune commands
                while let Ok(msg) = self.ctrl_rx.try_recv() {
                    match msg {
                        ControlMsg::RetuneAbs(f) => {
                            // rebase: new world origin = f, camera teleports to world 0.
                            // works for any frequency, no wall-extent constraint.
                            self.initial_freq_hz = f;
                            self.camera.pos.x = 0.0;
                            store_f32(&self.shared_camera_x, 0.0);
                            let _ = self.rebase_tx.send(f);
                            // drop any spectrum rows captured before the rebase
                            while self.rx.try_recv().is_ok() {}
                            gfx.clear_history();
                            eprintln!("warp → {:.3} MHz (rebase)", f as f64 / 1e6);
                        }
                        ControlMsg::RetuneRel(d) => {
                            let delta_m = d as f32 / HZ_PER_METER;
                            self.camera.pos.x += delta_m;
                        }
                        ControlMsg::SetMode(_) => {
                            eprintln!("mode switching not supported in hallway");
                        }
                    }
                }

                self.camera.pos.x = self
                    .camera
                    .pos
                    .x
                    .clamp(-WALL_HALF_LEN_M + 1.0, WALL_HALF_LEN_M - 1.0);
                store_f32(&self.shared_camera_x, self.camera.pos.x);
                if let Err(e) = gfx.render(&self.rx, &self.camera) {
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
    let (tx, rx) = mpsc::sync_channel::<SpecRow>(8);

    let (mut producer, consumer) = RingBuffer::<f32>::new(16_384);
    for _ in 0..8_192 {
        let _ = producer.push(0.0);
    }
    let _audio_out = audio::start(consumer)?;

    // world 0 = the user's --freq. Camera starts there. LO sits at world +4m
    // (= SHIFT_HZ above world origin), the standard +SHIFT_HZ LO offset.
    let shared_cam_x = Arc::new(AtomicU32::new(0));
    store_f32(&shared_cam_x, 0.0);
    let shared_lo_world_x = Arc::new(AtomicU32::new(0));
    store_f32(&shared_lo_world_x, (SHIFT_HZ as f32) / HZ_PER_METER);

    let (rebase_tx, rebase_rx) = mpsc::channel::<u64>();

    spawn_radio(
        freq_hz,
        tx,
        producer,
        stop.clone(),
        shared_cam_x.clone(),
        shared_lo_world_x.clone(),
        rebase_rx,
    );

    // stdin warp/retune: same parser as waterfall ("89.5M", "+", "-", etc.)
    let (ctrl_tx, ctrl_rx) = mpsc::channel::<ControlMsg>();
    spawn_stdin_reader(ctrl_tx, "hall");

    let event_loop = EventLoop::new()?;
    let mut app = App {
        window: None,
        gfx: None,
        rx,
        ctrl_rx,
        rebase_tx,
        stop: stop.clone(),
        camera: Camera {
            pos: Vec3::new(0.0, HEAD_HEIGHT, 0.0),
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 60f32.to_radians(),
            aspect: 1.0,
        },
        input: Input::default(),
        last_frame: std::time::Instant::now(),
        shared_camera_x: shared_cam_x,
        shared_lo_world_x,
        initial_freq_hz: freq_hz,
    };
    event_loop.run_app(&mut app)?;
    stop.store(true, Ordering::Relaxed);
    Ok(())
}

struct Gfx {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    spec_tex: wgpu::Texture,
    lo_buf: wgpu::Buffer,
    cam_buf: wgpu::Buffer,
    vbuf: wgpu::Buffer,
    vbuf_count: u32,
    depth_view: wgpu::TextureView,
    write_row: u32,
}

impl Gfx {
    async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
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
                label: Some("vidman hallway device"),
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

        let spec_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hallway spectrum"),
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
        let spec_view = spec_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hallway sampler"),
            ..Default::default()
        });

        let cam_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hallway cam"),
            size: std::mem::size_of::<CamUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // one f32 per historical row: where the LO was when this row was captured
        let lo_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hallway lo_per_row"),
            size: (HISTORY as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hallway bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hallway bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cam_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&spec_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: lo_buf.as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hallway shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("hallway.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hallway pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let depth_format = wgpu::TextureFormat::Depth32Float;
        let depth_view = make_depth(&device, config.width, config.height, depth_format);

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hallway pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // walls visible from both sides
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: Default::default(),
            cache: None,
        });

        let verts = hallway_mesh(WALL_HALF_LEN_M * 2.0, HALLWAY_WIDTH, HALLWAY_HEIGHT);
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hallway vbuf"),
            size: (verts.len() * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vbuf, 0, bytemuck::cast_slice(&verts));
        let vbuf_count = verts.len() as u32;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            bind_group,
            spec_tex,
            lo_buf,
            cam_buf,
            vbuf,
            vbuf_count,
            depth_view,
            write_row: 0,
        })
    }

    /// Reset row bookkeeping. Existing spec_tex contents are left alone but
    /// lo_buf is filled with a sentinel so old rows render as dead zone.
    fn clear_history(&mut self) {
        self.write_row = 0;
        let sentinel = vec![1.0e6f32; HISTORY as usize];
        self.queue
            .write_buffer(&self.lo_buf, 0, bytemuck::cast_slice(&sentinel));
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.config.width = w.max(1);
        self.config.height = h.max(1);
        self.surface.configure(&self.device, &self.config);
        self.depth_view = make_depth(
            &self.device,
            self.config.width,
            self.config.height,
            wgpu::TextureFormat::Depth32Float,
        );
    }

    fn render(&mut self, rx: &Receiver<SpecRow>, camera: &Camera) -> anyhow::Result<()> {
        // drain any pending spectrum rows into the scrolling texture, and
        // record the LO each was captured at into lo_buf at the matching index
        while let Ok(row) = rx.try_recv() {
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.spec_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: self.write_row,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&row.samples),
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
            self.queue.write_buffer(
                &self.lo_buf,
                (self.write_row as u64) * 4,
                bytemuck::bytes_of(&row.lo_world_x),
            );
            self.write_row = (self.write_row + 1) % HISTORY;
        }

        let cam = CamUniform {
            view_proj: camera.view_proj().to_cols_array_2d(),
            write_row: self.write_row,
            history: HISTORY,
            db_min: DB_MIN,
            db_max: DB_MAX,
            band_half_width_m: BAND_HALF_WIDTH_M,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        self.queue
            .write_buffer(&self.cam_buf, 0, bytemuck::bytes_of(&cam));

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            wgpu::CurrentSurfaceTexture::Timeout => return Ok(()),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            other => anyhow::bail!("surface acquire failed: {other:?}"),
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hallway enc"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hallway pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.01,
                            g: 0.00,
                            b: 0.04,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: Default::default(),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vbuf.slice(..));
            pass.draw(0..self.vbuf_count, 0..1);
        }
        self.queue.submit([enc.finish()]);
        frame.present();
        Ok(())
    }
}

fn make_depth(
    device: &wgpu::Device,
    w: u32,
    h: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hallway depth"),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Build the wall mesh: two long quads (left/right walls) facing inward.
/// Origin = hallway center; X along length, Y up, Z across width.
fn hallway_mesh(length: f32, width: f32, height: f32) -> Vec<Vertex> {
    let hl = length * 0.5;
    let hw = width * 0.5;
    let mut v = Vec::with_capacity(12);

    // right wall (z = +hw), normal facing -Z (interior side)
    let r00 = Vertex { pos: [-hl, 0.0, hw], uv: [0.0, 0.0] };
    let r10 = Vertex { pos: [ hl, 0.0, hw], uv: [1.0, 0.0] };
    let r11 = Vertex { pos: [ hl, height, hw], uv: [1.0, 1.0] };
    let r01 = Vertex { pos: [-hl, height, hw], uv: [0.0, 1.0] };
    v.extend_from_slice(&[r00, r10, r11, r00, r11, r01]);

    // left wall (z = -hw), normal facing +Z. cull_mode=None so winding doesn't matter
    let l00 = Vertex { pos: [-hl, 0.0, -hw], uv: [0.0, 0.0] };
    let l10 = Vertex { pos: [ hl, 0.0, -hw], uv: [1.0, 0.0] };
    let l11 = Vertex { pos: [ hl, height, -hw], uv: [1.0, 1.0] };
    let l01 = Vertex { pos: [-hl, height, -hw], uv: [0.0, 1.0] };
    v.extend_from_slice(&[l00, l11, l10, l00, l01, l11]);

    v
}

fn spawn_radio(
    freq_hz: u64,
    tx: SyncSender<SpecRow>,
    mut producer: Producer<f32>,
    stop: Arc<AtomicBool>,
    shared_cam_x: Arc<AtomicU32>,
    shared_lo_world_x: Arc<AtomicU32>,
    rebase_rx: Receiver<u64>,
) {
    std::thread::spawn(move || {
        let hackrf = HackRf::open_first().expect("open hackrf");
        let mut world_origin_hz: u64 = freq_hz;
        let mut current_lo_hz: u64 = world_origin_hz + SHIFT_HZ;
        let lo_offset_m: f32 = (SHIFT_HZ as f32) / HZ_PER_METER;
        let mut lo_world_x: f32 = lo_offset_m;
        hackrf
            .start_rx(&Config {
                txvga_db: 0,
                vga_db: 16,
                lna_db: 16,
                amp_enable: false,
                antenna_enable: false,
                frequency_hz: current_lo_hz,
                sample_rate_hz: RADIO_SAMPLE_RATE_HZ,
                sample_rate_div: 1,
            })
            .expect("start_rx");

        // single FM demod chain — same shape as waterfall::spawn_radio.
        // mixer is fixed at +SHIFT_HZ relative to LO; we retune the LO itself
        // to follow the camera, so the mixer never needs to change.
        let mut pipeline = Pipeline::fm();

        let mut spec = Spectrum::new(FFT_SIZE);
        let mut buf = vec![0u8; 262_144];
        let mut iq: Vec<Complex32> = Vec::with_capacity(buf.len() / 2);
        let mut mixed: Vec<Complex32> = Vec::new();
        let mut decimated: Vec<f32> = Vec::new();
        let mut audio_hi: Vec<f32> = Vec::new();
        let mut deemphed: Vec<f32> = Vec::new();
        let mut pushed = 0usize;
        let mut iters: u64 = 0;

        while !stop.load(Ordering::Relaxed) {
            // handle any pending rebases from main thread (instant warps)
            while let Ok(new_freq) = rebase_rx.try_recv() {
                let new_lo_hz = new_freq + SHIFT_HZ;
                match hackrf.set_freq(new_lo_hz) {
                    Ok(()) => {
                        world_origin_hz = new_freq;
                        current_lo_hz = new_lo_hz;
                        lo_world_x = lo_offset_m;
                        store_f32(&shared_lo_world_x, lo_world_x);
                    }
                    Err(e) => eprintln!("rebase retune error: {e:#}"),
                }
            }

            let n = match hackrf.read(&mut buf) {
                Ok(n) => n,
                Err(e) => {
                    eprintln!("hackrf read error: {e:#}");
                    continue;
                }
            };
            iq.clear();
            for chunk in buf[..n].chunks_exact(2) {
                let i = (chunk[0] as i8) as f32 / 128.0;
                let q = (chunk[1] as i8) as f32 / 128.0;
                iq.push(Complex32::new(i, q));
            }

            // retune the LO to follow the camera. target LO sits SHIFT_HZ above
            // the camera's world freq so audio stays clear of DC leakage.
            let cam_world_x = load_f32(&shared_cam_x);
            let target_lo_world_x = cam_world_x + lo_offset_m;
            if (target_lo_world_x - lo_world_x).abs() > RETUNE_THRESHOLD_M {
                let new_lo_hz = (world_origin_hz as f64
                    + (target_lo_world_x as f64) * (HZ_PER_METER as f64))
                    as u64;
                match hackrf.set_freq(new_lo_hz) {
                    Ok(()) => {
                        current_lo_hz = new_lo_hz;
                        lo_world_x = target_lo_world_x;
                        store_f32(&shared_lo_world_x, lo_world_x);
                    }
                    Err(e) => eprintln!("retune error: {e:#}"),
                }
            }

            // audio: demod whatever the LO is on; same call shape as waterfall
            pipeline.process(&iq, &mut mixed, &mut decimated, &mut audio_hi, &mut deemphed);
            for &s in &deemphed {
                let _ = producer.push(s);
            }

            iters += 1;
            if iters.is_multiple_of(18) {
                let cam_freq = (current_lo_hz as f64) - (SHIFT_HZ as f64);
                eprintln!(
                    "cam={:+.1}m → tuned {:.3} MHz",
                    cam_world_x,
                    cam_freq / 1e6
                );
            }

            // FFT path: feed the wall texture
            let mut s_idx = 0;
            while s_idx + FFT_SIZE <= iq.len() {
                spec.push(&iq[s_idx..s_idx + FFT_SIZE]);
                pushed += 1;
                s_idx += FFT_SIZE;
                if pushed == AVG_FFTS {
                    let mut row = spec.take();
                    notch_dc(&mut row);
                    if tx
                        .send(SpecRow {
                            samples: row,
                            lo_world_x,
                        })
                        .is_err()
                    {
                        let _ = hackrf.stop();
                        return;
                    }
                    pushed = 0;
                }
            }
        }
        let _ = hackrf.stop();
    });
}

// share with waterfall::notch_dc — both want the LO leakage smoothed out
fn notch_dc(row: &mut [f32]) {
    let mid = row.len() / 2;
    if mid < 2 || mid + 2 >= row.len() {
        return;
    }
    let avg = 0.5 * (row[mid - 2] + row[mid + 2]);
    row[mid - 1] = avg;
    row[mid] = avg;
    row[mid + 1] = avg;
}
