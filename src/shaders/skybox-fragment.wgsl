struct FragmentInput {
    @location(0) direction : vec3<f32>,
};

@group(1) @binding(0) var skyboxSampler: sampler;
@group(1) @binding(1) var skyboxData: texture_cube<f32>;

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    let color = textureSampleLevel(skyboxData, skyboxSampler, input.direction, 0).rgb;

    return vec4<f32>(color, 1.0);
}
