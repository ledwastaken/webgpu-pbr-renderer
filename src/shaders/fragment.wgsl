struct FragmentInput {
    @location(0) world_pos : vec3<f32>,
    @location(1) normal    : vec3<f32>,
    @location(2) uv        : vec2<f32>,
    @location(3) camera_pos: vec3<f32>,
};

@group(1) @binding(0) var baseColorSampler: sampler;
@group(1) @binding(1) var baseColorData: texture_2d<f32>;
@group(1) @binding(2) var normalSampler: sampler;
@group(1) @binding(3) var normalData: texture_2d<f32>;

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    let light_pos = vec3<f32>(0, 4, 0);

    let V = normalize(input.camera_pos - input.world_pos);
    let L = normalize(light_pos - input.world_pos);
    let H = normalize(V + L);

    let PI: f32 = 3.14159265359;

    let roughness = 0.0;

    let base_color = textureSample(baseColorData, baseColorSampler, input.uv);
    let normal_map = textureSample(normalData, normalSampler, input.uv);

    let rgb_normal = vec3<f32>(0.5, 0.5, 1);//normal_map * 2.0 - 1.0;
    // let T = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), input.normal));
    // let B = cross(input.normal, T);
    // let N = normalize(T * normal_map.x + B * normal_map.y + input.normal * normal_map.z);

    // let NdotH = max(dot(N, H), 0.0);
    // let NdotV = max(dot(N, V), 0.0);
    // let NdotL = max(dot(N, L), 0.0);
    // let VdotH = max(dot(V, H), 0.0);

    // let shininess = 100.0;
    // let R = reflect(-L, N);
    // let specular = pow(max(dot(R, V), 0.0), shininess) * vec3<f32>(1.0);

    // let diffuse = base_color.rgb * NdotL;

    // let finalColor = diffuse + specular * 0.5;
    let finalColor = normal_map.rgb;
    return vec4<f32>(finalColor.rgb, 1.0);
}
