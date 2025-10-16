(function(){const i=document.createElement("link").relList;if(i&&i.supports&&i.supports("modulepreload"))return;for(const e of document.querySelectorAll('link[rel="modulepreload"]'))s(e);new MutationObserver(e=>{for(const t of e)if(t.type==="childList")for(const u of t.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&s(u)}).observe(document,{childList:!0,subtree:!0});function n(e){const t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),e.crossOrigin==="use-credentials"?t.credentials="include":e.crossOrigin==="anonymous"?t.credentials="omit":t.credentials="same-origin",t}function s(e){if(e.ep)return;e.ep=!0;const t=n(e);fetch(e.href,t)}})();const N=`struct VertexInput {
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
`;function q(o=1,i=32,n=32){const e=[],t=[];for(let u=0;u<=i;u++){const p=u/i,c=p*Math.PI;for(let a=0;a<=n;a++){const y=a/n,v=y*Math.PI*2,l=Math.sin(c),f=Math.cos(c),d=Math.sin(v),w=Math.cos(v),m=o*l*w,g=o*f,h=o*l*d,P=m/o,B=g/o,G=h/o;e.push(m,g,h,P,B,G,y,1-p)}}for(let u=0;u<i;u++)for(let p=0;p<n;p++){const c=u*(n+1)+p,a=c+n+1;t.push(c,a,c+1),t.push(a,a+1,c+1)}return{vertices:new Float32Array(e),indices:new Uint32Array(t),vertexStride:32}}let x,r,U,A,_,O,T,M,b,I,W=performance.now(),z,S,C;const R=q(),F=R.vertices,E=R.indices;async function Y(){x=document.getElementById("GLCanvas");const o=await navigator.gpu?.requestAdapter();if(!o)throw new Error("Browser does not support WebGPU");if(r=await o?.requestDevice(),!r)throw new Error("Browser does not support WebGPU");if(U=x.getContext("webgpu"),!U)throw new Error("Failed to get canvas context");const i=window.devicePixelRatio||1;x.width=x.clientWidth*i,x.height=x.clientHeight*i,A=navigator.gpu.getPreferredCanvasFormat(),U.configure({device:r,format:A}),O=r.createTexture({size:[x.width,x.height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}),T=r.createBuffer({size:F.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(T.getMappedRange()).set(F),T.unmap(),M=r.createBuffer({size:E.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Uint16Array(M.getMappedRange()).set(E),M.unmap(),b=r.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const n="http://localhost:5173/webgpu-pbr-renderer/texture/Metal046B_2K-JPG_Color.jpg",s=document.createElement("img");s.src=n,await s.decode();const e=await createImageBitmap(s);z=r.createTexture({size:[e.width,e.height,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}),S=r.createSampler({magFilter:"linear",minFilter:"linear"}),r.queue.copyExternalImageToTexture({source:e},{texture:z},[e.width,e.height]);const t=r.createPipelineLayout({bindGroupLayouts:[r.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}}]}),r.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{}}]})]});_=r.createRenderPipeline({layout:t,vertex:{module:r.createShaderModule({code:N}),entryPoint:"main",buffers:[{arrayStride:32,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x2"}]}]},fragment:{module:r.createShaderModule({code:D}),entryPoint:"main",targets:[{format:A}]},primitive:{topology:"triangle-list",cullMode:"front"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),I=r.createBindGroup({layout:_.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:b}}]}),C=r.createBindGroup({layout:_.getBindGroupLayout(1),entries:[{binding:0,resource:S},{binding:1,resource:z.createView()}]})}function V(){const o=(performance.now()-W)/1e3,i=800/600,n=Math.PI/2,t=j(n,i,.1,100),u=X([1.5,0,1.5],[0,0,0],[0,1,0]),p=H(o*.5);r.queue.writeBuffer(b,0,new Float32Array(p)),r.queue.writeBuffer(b,64,new Float32Array(u)),r.queue.writeBuffer(b,128,new Float32Array(t));const c=r.createCommandEncoder(),a=c.beginRenderPass({colorAttachments:[{view:U.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:O.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});a.setPipeline(_),a.setVertexBuffer(0,T),a.setIndexBuffer(M,"uint16"),a.setBindGroup(0,I),a.setBindGroup(1,C),a.drawIndexed(E.length),a.end(),r.queue.submit([c.finish()]),requestAnimationFrame(V)}function j(o,i,n,s){const e=1/Math.tan(o/2),t=1/(n-s);return new Float32Array([e/i,0,0,0,0,e,0,0,0,0,(s+n)*t,-1,0,0,2*s*n*t,0])}function X(o,i,n){const[s,e,t]=o,[u,p,c]=i,[a,y,v]=n;let l=s-u,f=e-p,d=t-c;const w=Math.hypot(l,f,d);l/=w,f/=w,d/=w;let m=y*d-v*f,g=v*l-a*d,h=a*f-y*l;const P=Math.hypot(m,g,h);m/=P,g/=P,h/=P;const B=f*h-d*g,G=d*m-l*h,L=l*g-f*m;return new Float32Array([m,B,l,0,g,G,f,0,h,L,d,0,-(m*s+g*e+h*t),-(B*s+G*e+L*t),-(l*s+f*e+d*t),1])}function H(o){const i=Math.cos(o),n=Math.sin(o);return new Float32Array([i,0,n,0,0,1,0,0,-n,0,i,0,0,0,0,1])}Y().then(V);
