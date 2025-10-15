struct FragmentInput {
    @location(0) world_pos : vec3<f32>,
    @location(1) normal    : vec3<f32>,
    @location(2) uv        : vec2<f32>,
};

@group(1) @binding(0) var textureSampler: sampler;
@group(1) @binding(1) var textureData: texture_2d<f32>;

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    // let color = input.world_pos;
    let color = textureSample(textureData, textureSampler, input.uv);

    return vec4<f32>(color.rgb, 1.0);
}
