struct VertexInput {
  @location(0) position : vec4<f32>,
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

fn extract_rotation(m: mat4x4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(
        vec4<f32>(m[0].xyz, 0.0),
        vec4<f32>(m[1].xyz, 0.0),
        vec4<f32>(m[2].xyz, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );
}

@vertex
fn main(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  output.position = uniforms.proj * extract_rotation(uniforms.view) * input.position;
  output.normal = normalize(input.position.xyz);
  
  return output;
}
