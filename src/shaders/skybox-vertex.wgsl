struct VertexInput {
  @location(0) position : vec3<f32>,
};

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) normal : vec3<f32>,
};

struct Uniforms {
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.position = uniforms.proj * uniforms.view * vec4<f32>(input.position, 1.0);
  output.normal = normalize(input.position);
  
  return output;
}
