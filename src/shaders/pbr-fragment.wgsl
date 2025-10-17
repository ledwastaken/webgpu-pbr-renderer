struct FragmentInput {
    @location(0) world_pos : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) tangent : vec3<f32>,
    @location(4) bitangent : vec3<f32>,
};

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    return vec4<f32>(0.8, 0.5, 0.5, 1.0);
}
