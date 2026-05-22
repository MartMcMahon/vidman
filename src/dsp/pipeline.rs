use num_complex::Complex32;

use crate::dsp::{
    deemph::Deemph,
    demod::FmDemod,
    fir::{Fir, RealFir},
    mixer::Mixer,
};

pub enum Pipeline {
    Fm(FmPipeline),
    Gmrs(GmrsPipeline),
}

pub struct FmPipeline {
    pub mixer: Mixer,
    pub fir1: Fir, // 2.4M -> 240k LP:100k
    pub fm: FmDemod,
    pub fir2: RealFir, // 240k -> 48k(for audio) LP:15k
    pub deemph: Deemph,
}

pub struct GmrsPipeline {
    pub mixer: Mixer,
    pub fir1: Fir, // 2.4M -> 240k LP:100k
    pub fir2: Fir, // 240k -> 48k LP:6.25k
    pub fm: FmDemod,
    pub fir_audio: RealFir, // 48k -> 48k LP:3k
}

impl Pipeline {
    pub fn fm() -> Self {
        Self::Fm(FmPipeline {
            mixer: Mixer::new(200_000.0, 2_400_000.0),
            fir1: Fir::new(63, 100_000.0, 2_400_000.0, 10),
            fm: FmDemod::new(),
            fir2: RealFir::new(63, 15_000.0, 240_000.0, 5),
            deemph: Deemph::new(75e-6, 48_000.0),
        })
    }

    pub fn gmrs() -> Self {
        Self::Gmrs(GmrsPipeline {
            mixer: Mixer::new(200_000.0, 2_400_000.0),
            fir1: Fir::new(63, 100_000.0, 2_400_000.0, 10),
            fir2: Fir::new(63, 6_250.0, 240_000.0, 5),
            fm: FmDemod::new(),
            fir_audio: RealFir::new(63, 3_000.0, 48_000.0, 1),
        })
    }

    pub fn process(
        &mut self,
        iq: &[Complex32],
        scratch_complex: &mut Vec<Complex32>,
        scratch_real_a: &mut Vec<f32>,
        scratch_real_b: &mut Vec<f32>,
        out: &mut Vec<f32>,
    ) {
        match self {
            Pipeline::Fm(pipeline) => {
                scratch_complex.clear();
                scratch_real_a.clear();
                scratch_real_b.clear();
                out.clear();

                for &z in iq {
                    scratch_complex.push(pipeline.mixer.mix(z));
                }
                let mut decimated = Vec::new();
                pipeline.fir1.process(scratch_complex, &mut decimated);
                pipeline.fm.process(&decimated, scratch_real_a);
                pipeline.fir2.process(scratch_real_a, scratch_real_b);
                pipeline.deemph.process(scratch_real_b, out);
            }
            Pipeline::Gmrs(pipeline) => {
                scratch_complex.clear();
                scratch_real_a.clear();
                out.clear();

                for &z in iq {
                    scratch_complex.push(pipeline.mixer.mix(z));
                }

                let mut decimated1 = Vec::new();
                let mut decimated2 = Vec::new();
                pipeline.fir1.process(scratch_complex, &mut decimated1);
                pipeline.fir2.process(&decimated1, &mut decimated2);
                pipeline.fm.process(&decimated2, scratch_real_a);
                pipeline.fir_audio.process(scratch_real_a, out);
            }
        }
    }
}
