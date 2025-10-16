(function(){const i=document.createElement("link").relList;if(i&&i.supports&&i.supports("modulepreload"))return;for(const e of document.querySelectorAll('link[rel="modulepreload"]'))s(e);new MutationObserver(e=>{for(const t of e)if(t.type==="childList")for(const u of t.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&s(u)}).observe(document,{childList:!0,subtree:!0});function r(e){const t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),e.crossOrigin==="use-credentials"?t.credentials="include":e.crossOrigin==="anonymous"?t.credentials="omit":t.credentials="same-origin",t}function s(e){if(e.ep)return;e.ep=!0;const t=r(e);fetch(e.href,t)}})();const N=`struct VertexInput {
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
`,D=`struct FragmentInput {
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
`;function q(o=1,i=32,r=32){const e=[],t=[];for(let u=0;u<=i;u++){const p=u/i,c=p*Math.PI;for(let a=0;a<=r;a++){const y=a/r,v=y*Math.PI*2,l=Math.sin(c),f=Math.cos(c),d=Math.sin(v),w=Math.cos(v),m=o*l*w,h=o*f,g=o*l*d,P=m/o,b=h/o,G=g/o;e.push(m,h,g,P,b,G,y,1-p)}}for(let u=0;u<i;u++)for(let p=0;p<r;p++){const c=u*(r+1)+p,a=c+r+1;t.push(c,a,c+1),t.push(a,a+1,c+1)}return{vertices:new Float32Array(e),indices:new Uint32Array(t),vertexStride:32}}let x,n,U,M,_,O,T,A,B,I,W=performance.now(),z,S,C;const R=q(),F=R.vertices,E=R.indices;async function Y(){x=document.getElementById("GLCanvas");const o=await navigator.gpu?.requestAdapter();if(!o)throw new Error("Browser does not support WebGPU");if(n=await o?.requestDevice(),!n)throw new Error("Browser does not support WebGPU");if(U=x.getContext("webgpu"),!U)throw new Error("Failed to get canvas context");const i=window.devicePixelRatio||1;x.width=x.clientWidth*i,x.height=x.clientHeight*i,M=navigator.gpu.getPreferredCanvasFormat(),U.configure({device:n,format:M}),O=n.createTexture({size:[x.width,x.height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}),T=n.createBuffer({size:F.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(T.getMappedRange()).set(F),T.unmap(),A=n.createBuffer({size:E.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Uint16Array(A.getMappedRange()).set(E),A.unmap(),B=n.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const r="http://localhost:5173/texture/Rock058_8K-JPG_Color.jpg",s=document.createElement("img");s.src=r,await s.decode();const e=await createImageBitmap(s);z=n.createTexture({size:[e.width,e.height,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}),S=n.createSampler({magFilter:"linear",minFilter:"linear"}),n.queue.copyExternalImageToTexture({source:e},{texture:z},[e.width,e.height]);const t=n.createPipelineLayout({bindGroupLayouts:[n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}}]}),n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{}}]})]});_=n.createRenderPipeline({layout:t,vertex:{module:n.createShaderModule({code:N}),entryPoint:"main",buffers:[{arrayStride:32,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x2"}]}]},fragment:{module:n.createShaderModule({code:D}),entryPoint:"main",targets:[{format:M}]},primitive:{topology:"triangle-list",cullMode:"front"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),I=n.createBindGroup({layout:_.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:B}}]}),C=n.createBindGroup({layout:_.getBindGroupLayout(1),entries:[{binding:0,resource:S},{binding:1,resource:z.createView()}]})}function V(){const o=(performance.now()-W)/1e3,i=800/600,r=Math.PI/2,t=j(r,i,.1,100),u=X([1.5,0,1.5],[0,0,0],[0,1,0]),p=H(o*.5);n.queue.writeBuffer(B,0,new Float32Array(p)),n.queue.writeBuffer(B,64,new Float32Array(u)),n.queue.writeBuffer(B,128,new Float32Array(t));const c=n.createCommandEncoder(),a=c.beginRenderPass({colorAttachments:[{view:U.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:O.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});a.setPipeline(_),a.setVertexBuffer(0,T),a.setIndexBuffer(A,"uint16"),a.setBindGroup(0,I),a.setBindGroup(1,C),a.drawIndexed(E.length),a.end(),n.queue.submit([c.finish()]),requestAnimationFrame(V)}function j(o,i,r,s){const e=1/Math.tan(o/2),t=1/(r-s);return new Float32Array([e/i,0,0,0,0,e,0,0,0,0,(s+r)*t,-1,0,0,2*s*r*t,0])}function X(o,i,r){const[s,e,t]=o,[u,p,c]=i,[a,y,v]=r;let l=s-u,f=e-p,d=t-c;const w=Math.hypot(l,f,d);l/=w,f/=w,d/=w;let m=y*d-v*f,h=v*l-a*d,g=a*f-y*l;const P=Math.hypot(m,h,g);m/=P,h/=P,g/=P;const b=f*g-d*h,G=d*m-l*g,L=l*h-f*m;return new Float32Array([m,b,l,0,h,G,f,0,g,L,d,0,-(m*s+h*e+g*t),-(b*s+G*e+L*t),-(l*s+f*e+d*t),1])}function H(o){const i=Math.cos(o),r=Math.sin(o);return new Float32Array([i,0,r,0,0,1,0,0,-r,0,i,0,0,0,0,1])}Y().then(V);
