(function(){const i=document.createElement("link").relList;if(i&&i.supports&&i.supports("modulepreload"))return;for(const e of document.querySelectorAll('link[rel="modulepreload"]'))s(e);new MutationObserver(e=>{for(const t of e)if(t.type==="childList")for(const u of t.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&s(u)}).observe(document,{childList:!0,subtree:!0});function r(e){const t={};return e.integrity&&(t.integrity=e.integrity),e.referrerPolicy&&(t.referrerPolicy=e.referrerPolicy),e.crossOrigin==="use-credentials"?t.credentials="include":e.crossOrigin==="anonymous"?t.credentials="omit":t.credentials="same-origin",t}function s(e){if(e.ep)return;e.ep=!0;const t=r(e);fetch(e.href,t)}})();const R=`struct VertexInput {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position : vec4<f32>,
    @location(0) world_pos : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) camera_pos: vec3<f32>,
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

fn camera_pos_from_view(view: mat4x4<f32>) -> vec3<f32> {
    return vec3<f32>(
        -dot(view[0].xyz, view[3].xyz),
        -dot(view[1].xyz, view[3].xyz),
        -dot(view[2].xyz, view[3].xyz)
    );
}

@vertex
fn main(input: VertexInput) -> VertexOutput {
    let world_pos = uniforms.model * vec4<f32>(input.position, 1.0);
    let normal_world = normalize(mat3_from_mat4(uniforms.model) * input.normal);
    
    var output: VertexOutput;
    output.clip_position = uniforms.proj * uniforms.view * world_pos;
    output.world_pos = world_pos.xyz;
    output.normal = normal_world;
    output.uv = input.uv;
    output.camera_pos = camera_pos_from_view(uniforms.view);
    
    return output;
}`,D=`struct FragmentInput {
    @location(0) world_pos : vec3<f32>,
    @location(1) normal    : vec3<f32>,
    @location(2) uv        : vec2<f32>,
    @location(3) camera_pos: vec3<f32>,
};

@group(1) @binding(0) var textureSampler: sampler;
@group(1) @binding(1) var textureData: texture_2d<f32>;

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    let light_pos = vec3<f32>(20, 10, 4);
    let N = normalize(input.normal);
    let V = normalize(input.camera_pos - input.world_pos);
    let L = normalize(light_pos - input.world_pos);
    let H = normalize(V + L);
    let PI: f32 = 3.14159265359;

    let base_color = textureSample(textureData, textureSampler, input.uv);

    let NdotH = max(dot(N, H), 0.0);
    let NdotV = max(dot(N, V), 0.0);
    let NdotL = max(dot(N, L), 0.0);
    let VdotH = max(dot(V, H), 0.0);

    let shininess = 100.0;
    let R = reflect(-L, N);
    let specular = pow(max(dot(R, V), 0.0), shininess) * vec3<f32>(1.0);

    let diffuse = base_color.rgb * NdotL;

    let finalColor = diffuse + specular * 0.5;
    return vec4<f32>(finalColor.rgb, 1.0);
}
`;function q(o=1,i=32,r=32){const e=[],t=[];for(let u=0;u<=i;u++){const f=u/i,c=f*Math.PI;for(let a=0;a<=r;a++){const v=a/r,y=v*Math.PI*2,l=Math.sin(c),p=Math.cos(c),d=Math.sin(y),w=Math.cos(y),m=o*l*w,x=o*p,g=o*l*d,_=m/o,P=x/o,B=g/o;e.push(m,x,g,_,P,B,v,1-f)}}for(let u=0;u<i;u++)for(let f=0;f<r;f++){const c=u*(r+1)+f,a=c+r+1;t.push(c,a,c+1),t.push(a,a+1,c+1)}return{vertices:new Float32Array(e),indices:new Uint32Array(t),vertexStride:32}}let h,n,z,T,G,V,U,L,b,F,H=performance.now(),M,E,I;const O=q(),N=O.vertices,A=O.indices;async function W(){h=document.getElementById("GLCanvas");const o=await navigator.gpu?.requestAdapter();if(!o)throw new Error("Browser does not support WebGPU");if(n=await o?.requestDevice(),!n)throw new Error("Browser does not support WebGPU");if(z=h.getContext("webgpu"),!z)throw new Error("Failed to get canvas context");const i=window.devicePixelRatio||1;h.width=h.clientWidth*i,h.height=h.clientHeight*i,T=navigator.gpu.getPreferredCanvasFormat(),z.configure({device:n,format:T}),V=n.createTexture({size:[h.width,h.height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}),U=n.createBuffer({size:N.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(U.getMappedRange()).set(N),U.unmap(),L=n.createBuffer({size:A.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Uint16Array(L.getMappedRange()).set(A),L.unmap(),b=n.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});const r="/webgpu-pbr-renderer/texture/Metal046B_2K-JPG_Color.jpg",s=document.createElement("img");s.src=r,await s.decode();const e=await createImageBitmap(s);M=n.createTexture({size:[e.width,e.height,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}),E=n.createSampler({magFilter:"linear",minFilter:"linear"}),n.queue.copyExternalImageToTexture({source:e},{texture:M},[e.width,e.height]);const t=n.createPipelineLayout({bindGroupLayouts:[n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}}]}),n.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{}}]})]});G=n.createRenderPipeline({layout:t,vertex:{module:n.createShaderModule({code:R}),entryPoint:"main",buffers:[{arrayStride:32,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x2"}]}]},fragment:{module:n.createShaderModule({code:D}),entryPoint:"main",targets:[{format:T}]},primitive:{topology:"triangle-list",cullMode:"front"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),F=n.createBindGroup({layout:G.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:b}}]}),I=n.createBindGroup({layout:G.getBindGroupLayout(1),entries:[{binding:0,resource:E},{binding:1,resource:M.createView()}]})}function C(){const o=(performance.now()-H)/1e3,i=800/600,r=Math.PI/2,t=Y(r,i,.1,100),u=j([1.5,0,1.5],[0,0,0],[0,1,0]),f=X(o*.3);n.queue.writeBuffer(b,0,new Float32Array(f)),n.queue.writeBuffer(b,64,new Float32Array(u)),n.queue.writeBuffer(b,128,new Float32Array(t));const c=n.createCommandEncoder(),a=c.beginRenderPass({colorAttachments:[{view:z.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:V.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});a.setPipeline(G),a.setVertexBuffer(0,U),a.setIndexBuffer(L,"uint16"),a.setBindGroup(0,F),a.setBindGroup(1,I),a.drawIndexed(A.length),a.end(),n.queue.submit([c.finish()]),requestAnimationFrame(C)}function Y(o,i,r,s){const e=1/Math.tan(o/2),t=1/(r-s);return new Float32Array([e/i,0,0,0,0,e,0,0,0,0,(s+r)*t,-1,0,0,2*s*r*t,0])}function j(o,i,r){const[s,e,t]=o,[u,f,c]=i,[a,v,y]=r;let l=s-u,p=e-f,d=t-c;const w=Math.hypot(l,p,d);l/=w,p/=w,d/=w;let m=v*d-y*p,x=y*l-a*d,g=a*p-v*l;const _=Math.hypot(m,x,g);m/=_,x/=_,g/=_;const P=p*g-d*x,B=d*m-l*g,S=l*x-p*m;return new Float32Array([m,P,l,0,x,B,p,0,g,S,d,0,-(m*s+x*e+g*t),-(P*s+B*e+S*t),-(l*s+p*e+d*t),1])}function X(o){const i=Math.cos(o),r=Math.sin(o);return new Float32Array([i,0,r,0,0,1,0,0,-r,0,i,0,0,0,0,1])}W().then(C);
