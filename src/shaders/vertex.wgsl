struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) normal   : vec3<f32>,
  @location(2) uv       : vec2<f32>,
};

struct VertexOutput {
  @builtin(position) clip_position : vec4<f32>,
  @location(0) world_pos : vec3<f32>,
  @location(1) normal    : vec3<f32>,
  @location(2) uv        : vec2<f32>,
};

struct Uniforms {
  model: mat4x4<f32>,
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

fn mat3_from_mat4(m: mat4x4<f32>) -> mat3x3<f32> {
  return mat3x3<f32>(
    m[0].xyz,
    m[1].xyz,
    m[2].xyz,
  );
}

@vertex
fn main(input: VertexInput) -> VertexOutput {
  let world_pos = uniforms.model * vec4<f32>(input.position, 1.0);
  let normal_world = normalize(mat3_from_mat4(uniforms.model) * input.normal);

  var output: VertexOutput;
  output.clip_position = uniforms.proj * uniforms.view * world_pos;
  output.world_pos = input.position;
  output.normal = normal_world;
  output.uv = input.uv;

  return output;
}
