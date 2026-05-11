use crate::linear_split;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Source};
use objc2_metal::{MTLResourceUsage, MTLSize};

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let num_dims = shape.len();
    let work_per_threadgroup = length / out_length;

    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    set_params!(
        encoder,
        (
            length as u32,
            num_dims as u32,
            shape.as_slice(),
            work_per_threadgroup as u32,
            &input,
            (output, output_offset)
        )
    );

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_reduce_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let num_dims = shape.len();
    let work_per_threadgroup = length / out_length;

    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    let strides: Vec<u32> = strides.iter().map(|&x| x as u32).collect();
    set_params!(
        encoder,
        (
            length as u32,
            num_dims as u32,
            shape.as_slice(),
            strides.as_slice(),
            work_per_threadgroup as u32,
            &input,
            (output, output_offset)
        )
    );

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_argmax_suppressed_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    vocab_size: usize,
    input: BufferOffset,
    suppress_ids: &Buffer,
    suppress_len: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, "argmax_suppressed_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (&input, suppress_ids, (output, output_offset), vocab_size as u32, suppress_len as u32)
    );

    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), 256).max(1);
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(suppress_ids, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_topk_suppressed_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    vocab_size: usize,
    k: usize,
    threads_per_tile: usize,
    tile_count: usize,
    input: BufferOffset,
    suppress_ids: &Buffer,
    suppress_len: usize,
    tile_values: &Buffer,
    tile_indices: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, "topk_suppressed_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            &input,
            suppress_ids,
            (tile_values, 0),
            (tile_indices, 0),
            vocab_size as u32,
            suppress_len as u32,
            k as u32
        )
    );

    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(suppress_ids, MTLResourceUsage::Read);
    encoder.use_resource(tile_values, MTLResourceUsage::Write);
    encoder.use_resource(tile_indices, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: tile_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_tile,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_last_softmax(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let work_per_threadgroup = elements;

    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (length, work_per_threadgroup, (input, input_offset), (output, output_offset))
    );

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rms_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            (output, output_offset),
            (alpha, alpha_offset),
            eps
        )
    );
    let work_per_threadgroup = elements_to_sum;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(alpha, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_add_rms_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_per_block: usize,
    eps: f32,
    input_a: &Buffer,
    input_a_offset: usize,
    input_b: &Buffer,
    input_b_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_per_block,
            (input_a, input_a_offset),
            (input_b, input_b_offset),
            (output, output_offset),
            (alpha, alpha_offset),
            eps
        )
    );
    let work_per_threadgroup = elements_per_block;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input_a, MTLResourceUsage::Read);
    encoder.use_resource(input_b, MTLResourceUsage::Read);
    encoder.use_resource(alpha, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    beta: &Buffer,
    beta_offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            length,
            elements_to_sum,
            (input, input_offset),
            (output, output_offset),
            (alpha, alpha_offset),
            (beta, beta_offset),
            eps
        )
    );

    let work_per_threadgroup = elements_to_sum;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.use_resource(input, MTLResourceUsage::Read);
    encoder.use_resource(alpha, MTLResourceUsage::Read);
    encoder.use_resource(beta, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_i(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            bh,
            td,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (output, output_offset)
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope_thd(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    b: usize,
    t: usize,
    h: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            b,
            t,
            h,
            d,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (output, output_offset)
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (b * t * h * d) / 2);
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_rope(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            bh,
            td,
            d,
            stride_b,
            (src, src_offset),
            (cos, cos_offset),
            (sin, sin_offset),
            (output, output_offset)
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.use_resource(src, MTLResourceUsage::Read);
    encoder.use_resource(cos, MTLResourceUsage::Read);
    encoder.use_resource(sin, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
