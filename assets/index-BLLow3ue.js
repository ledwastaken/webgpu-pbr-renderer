(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const n of i)if(n.type==="childList")for(const a of n.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&r(a)}).observe(document,{childList:!0,subtree:!0});function t(i){const n={};return i.integrity&&(n.integrity=i.integrity),i.referrerPolicy&&(n.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?n.credentials="include":i.crossOrigin==="anonymous"?n.credentials="omit":n.credentials="same-origin",n}function r(i){if(i.ep)return;i.ep=!0;const n=t(i);fetch(i.href,n)}})();class h{x;y;z;constructor(e=0,t=0,r=0){this.x=e,this.y=t,this.z=r}magnitude(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}unit(){const e=this.magnitude();return e===0?new h(0,0,0):new h(this.x/e,this.y/e,this.z/e)}cross(e){return new h(this.y*e.z-this.z*e.y,this.z*e.x-this.x*e.z,this.x*e.y-this.y*e.x)}negate(){return new h(-this.x,-this.y,-this.z)}add(e){return new h(this.x+e.x,this.y+e.y,this.z+e.z)}subtract(e){return new h(this.x-e.x,this.y-e.y,this.z-e.z)}multiply(e){return new h(this.x*e,this.y*e,this.z*e)}divide(e){if(e===0)throw new Error("Cannot divide by zero");return new h(this.x/e,this.y/e,this.z/e)}addAssign(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}subtractAssign(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}multiplyAssign(e){return this.x*=e,this.y*=e,this.z*=e,this}divideAssign(e){if(e===0)throw new Error("Cannot divide by zero");return this.x/=e,this.y/=e,this.z/=e,this}data(){return[this.x,this.y,this.z]}toString(){return`Vector3(${this.x}, ${this.y}, ${this.z})`}}class b{data_;constructor(e){if(e.length!==16)throw new Error("Matrix4 data array must have exactly 16 elements");this.data_=[...e]}static identity(){const e=new Array(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);return new b(e)}static frustum(e,t,r,i,n,a){const c=2*n,s=t-e,p=i-r,l=a-n,d=new Array(c/s,0,0,0,0,-c/p,0,0,(t+e)/s,(i+r)/p,-(a+n)/l,-1,0,0,-c*a/l,0);return new b(d)}static perspective(e,t,r,i){const n=Math.tan(e/2)*r,a=-n,c=n*t,s=-c;return this.frustum(s,c,a,n,r,i)}data(){return[...this.data_]}dataMutable(){return[...this.data_]}get(e,t){if(e<0||e>3||t<0||t>3)throw new Error("Matrix index out of bounds");return this.data_[e*4+t]}set(e,t,r){if(e<0||e>3||t<0||t>3)throw new Error("Matrix index out of bounds");this.data_[e*4+t]=r}toString(){const e=[];for(let t=0;t<4;t++){const r=this.data_.slice(t*4,(t+1)*4);e.push(`[${r.map(i=>i.toFixed(4)).join(", ")}]`)}return`Matrix4(
  ${e.join(`,
  `)}
)`}multiply(e){const t=this.data_,r=e.data_,i=new Array(16);for(let n=0;n<4;n++)for(let a=0;a<4;a++){let c=0;for(let s=0;s<4;s++)c+=t[n*4+s]*r[s*4+a];i[n*4+a]=c}return new b(i)}transpose(){const e=new Array(16);for(let t=0;t<4;t++)for(let r=0;r<4;r++)e[r*4+t]=this.data_[t*4+r];return new b(e)}}class S{r00=1;r01=0;r02=0;r10=0;r11=1;r12=0;r20=0;r21=0;r22=1;pos=new h;toMatrix(){const e=new Array(this.r00,this.r01,this.r02,this.pos.x,this.r10,this.r11,this.r12,this.pos.y,this.r20,this.r21,this.r22,this.pos.z,0,0,0,1);return new b(e)}invert(){let e=new S;return e.r00=this.r00,e.r01=this.r10,e.r02=this.r20,e.r10=this.r01,e.r11=this.r11,e.r12=this.r21,e.r20=this.r02,e.r21=this.r12,e.r22=this.r22,e.pos=new h(-(this.r00*this.pos.x+this.r01*this.pos.y+this.r02*this.pos.z),-(this.r10*this.pos.x+this.r11*this.pos.y+this.r12*this.pos.z),-(this.r20*this.pos.x+this.r21*this.pos.y+this.r22*this.pos.z)),e}add(e){let t=new S;return t.r00=this.r00,t.r01=this.r01,t.r02=this.r02,t.r10=this.r10,t.r11=this.r11,t.r12=this.r12,t.r20=this.r20,t.r21=this.r21,t.r22=this.r22,t.pos=this.pos.add(e),t}multiply(e){return new h(this.r00*e.x+this.r01*e.y+this.r02*e.z+this.pos.x,this.r10*e.x+this.r11*e.y+this.r12*e.z+this.pos.y,this.r20*e.x+this.r21*e.y+this.r22*e.z+this.pos.z)}data(){return[this.r00,this.r01,this.r02,this.r10,this.r11,this.r12,this.r20,this.r21,this.r22,this.pos.x,this.pos.y,this.pos.z]}getPosition(){return new h(this.pos.x,this.pos.y,this.pos.z)}getRight(){return new h(this.r00,this.r10,this.r20)}getUp(){return new h(this.r01,this.r11,this.r21)}getBack(){return new h(this.r02,this.r12,this.r22)}toString(){return`CFrame(pos: ${this.pos.toString()}, rotation: [${this.r00}, ${this.r01}, ${this.r02}, ${this.r10}, ${this.r11}, ${this.r12}, ${this.r20}, ${this.r21}, ${this.r22}])`}}class L{cframe;indicesCount;vertexBuffer;indexBuffer;constructor(e,t){this.cframe=new S,this.indicesCount=t.length,this.vertexBuffer=o.device.createBuffer({size:e.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.vertexBuffer.getMappedRange()).set(e),this.vertexBuffer.unmap(),this.indexBuffer=o.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Int16Array(this.indexBuffer.getMappedRange()).set(t),this.indexBuffer.unmap()}}const R=`struct VertexInput {
  @location(0) position : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) tangent : vec3<f32>,
};

struct VertexOutput {
  @builtin(position) clip_position : vec4<f32>,
  @location(0) world_pos : vec3<f32>,
  @location(1) normal : vec3<f32>,
  @location(2) uv : vec2<f32>,
  @location(3) tangent: vec3<f32>,
  @location(4) bitangent: vec3<f32>,
};

struct Uniforms {
  model: mat4x4<f32>,
  view: mat4x4<f32>,
  proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;

@vertex
fn main(input: VertexInput) -> VertexOutput {
  let world_pos = uniforms.model * vec4<f32>(input.position, 1.0);
  let normal_world = normalize((uniforms.model * vec4<f32>(input.normal, 0.0)).xyz);
  let tangent_world = normalize((uniforms.model * vec4<f32>(input.tangent, 0.0)).xyz);
  let bitangent_world = normalize(cross(normal_world, tangent_world));
  
  var output: VertexOutput;
  output.clip_position = uniforms.proj * uniforms.view * world_pos;
  output.world_pos = world_pos.xyz;
  output.normal = normal_world;
  output.uv = input.uv;
  output.tangent = tangent_world;
  output.bitangent = bitangent_world;
  
  return output;
}
`,k=`struct FragmentInput {
    @location(0) world_pos : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) tangent : vec3<f32>,
    @location(4) bitangent : vec3<f32>,
};

struct Uniforms {
    camera_pos: vec3<f32>,
    light_pos: vec3<f32>,
};

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

@group(2) @binding(0) var albedoSampler: sampler;
@group(2) @binding(1) var albedoData: texture_2d<f32>;
@group(2) @binding(2) var roughnessSampler: sampler;
@group(2) @binding(3) var roughnessData: texture_2d<f32>;

@group(3) @binding(0) var skyboxSampler: sampler;
@group(3) @binding(1) var skyboxData: texture_cube<f32>;

// GGX/Trowbridge-Reitz Normal Distribution Function
fn D(alpha: f32, N: vec3<f32>, H: vec3<f32>) -> f32 {
    const pi = 3.14159265359;
    let numerator = pow(alpha, 2.0);

    let NdotH = max(dot(N, H), 0.0);
    var denominator = pi * pow(pow(NdotH, 2.0) * (pow(alpha, 2.0) - 1.0) + 1.0, 2.0);
    denominator = max(denominator, 0.00001);

    return numerator / denominator;
}

// Schlick-Beckmann Geometry Shadowing Function
fn G1(alpha: f32, N: vec3<f32>, X: vec3<f32>) -> f32 {
    let NdotX = max(dot(N, X), 0.0);
    // For direct lighting, use this remapping:
    let k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
    let denominator = NdotX * (1.0 - k) + k;
    return NdotX / max(denominator, 0.00001);
}

// Smith model
fn G(alpha: f32, N: vec3<f32>, V: vec3<f32>, L: vec3<f32>) -> f32 {
    return G1(alpha, N, V) * G1(alpha, N, L);
}

// Fresnel-Schlick Function
fn F(F0: vec3<f32>, V: vec3<f32>, H: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(1 - max(dot(H, V), 0.0), 5.0);
}

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    const pi = 3.14159265359;
    let camera_pos = uniforms.camera_pos;
    let light_pos = uniforms.light_pos;
    let light_color = vec3<f32>(300.0);
    let albedo = textureSample(albedoData, albedoSampler, input.uv).rgb;
    let roughness = textureSample(roughnessData, roughnessSampler, input.uv).r;
    let metallic = 1.0;
    let emissivity = vec3<f32>(0.04);

    let F0 = mix(vec3<f32>(0.04), albedo, metallic);
    let alpha = roughness * roughness;

    let N = normalize(input.normal);
    let V = normalize(camera_pos - input.world_pos);
    let L = normalize(light_pos - input.world_pos);
    let H = normalize(V + L);
    let R = reflect(-V, N);

    let envColor = textureSample(skyboxData, skyboxSampler, R).rgb;

    // attenuation
    let distance = length(light_pos - input.world_pos);
    let attenuation = 1.0 / (distance * distance);
    let radiance = light_color * attenuation;

    let Ks = F(F0, V, H);
    let Kd = (vec3<f32>(1.0) - Ks) * (1.0 - metallic);
    let lambert = albedo / pi;

    let cookTorranceNumerator = D(alpha, N, H) * G(alpha, N, V, L) * F(F0, V, H);
    let cookTorranceDenominator = max(4.0 * max(dot(V, N), 0.0) * max(dot(L, N), 0.0), 0.0001);
    let cookTorrance = cookTorranceNumerator / cookTorranceDenominator;

    let specularIBL = envColor * Ks;

    let BRDF = Kd * lambert + cookTorrance;
    let outgoingLight = emissivity + BRDF * radiance * max(dot(L, N), 0.0) + specularIBL;

    return vec4<f32>(outgoingLight, 1.0);
}
`;class V{pipeline;uniformBuffer;uniformBindGroup;depthTexture;albedoSampler;albedoTexture;roughnessSampler;roughnessTexture;textureBindGroup;async init(){const e=o.device.createPipelineLayout({bindGroupLayouts:[o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}}]}),o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{}}]}),o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{viewDimension:"cube"}}]})]});this.pipeline=o.device.createRenderPipeline({layout:e,vertex:{module:o.device.createShaderModule({code:R}),entryPoint:"main",buffers:[{arrayStride:44,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x2"},{shaderLocation:3,offset:32,format:"float32x3"}]}]},fragment:{module:o.device.createShaderModule({code:k}),entryPoint:"main",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"front"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.uniformBuffer=o.device.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.uniformBindGroup=o.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),[this.roughnessTexture,this.roughnessSampler]=await this.loadTexture("texture/Metal046B_2K-JPG_Roughness.jpg"),[this.albedoTexture,this.albedoSampler]=await this.loadTexture("texture/Metal046B_2K-JPG_Color.jpg"),this.textureBindGroup=o.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(2),entries:[{binding:0,resource:this.albedoSampler},{binding:1,resource:this.albedoTexture.createView()},{binding:2,resource:this.roughnessSampler},{binding:3,resource:this.roughnessTexture.createView()}]}),this.depthTexture=o.device.createTexture({size:[800,600],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT})}async loadTexture(e){const t=document.createElement("img");t.src=e,await t.decode();let r=await createImageBitmap(t),i=o.device.createTexture({size:[r.width,r.height,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}),n=o.device.createSampler({magFilter:"linear",minFilter:"linear"});return o.device.queue.copyExternalImageToTexture({source:r},{texture:i},[r.width,r.height]),[i,n]}draw(e,t,r,i,n,a){const c=C(0);o.device.queue.writeBuffer(this.uniformBuffer,0,new Float32Array(c)),o.device.queue.writeBuffer(this.uniformBuffer,64,new Float32Array(r)),o.device.queue.writeBuffer(this.uniformBuffer,128,new Float32Array(i));const s=e.beginRenderPass({colorAttachments:[{view:o.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:this.depthTexture.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});s.setPipeline(this.pipeline),s.setVertexBuffer(0,t.vertexBuffer),s.setIndexBuffer(t.indexBuffer,"uint16"),s.setBindGroup(0,this.uniformBindGroup),s.setBindGroup(1,n),s.setBindGroup(2,this.textureBindGroup),s.setBindGroup(3,a),s.drawIndexed(t.indicesCount),s.end()}}let E=new V;function C(u){const e=Math.cos(u),t=Math.sin(u);return new Float32Array([e,0,t,0,0,1,0,0,-t,0,e,0,0,0,0,1])}const I=`struct VertexInput {
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
`,D=`struct FragmentInput {
    @location(0) direction : vec3<f32>,
};

@group(1) @binding(0) var skyboxSampler: sampler;
@group(1) @binding(1) var skyboxData: texture_cube<f32>;

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    let color = textureSample(skyboxData, skyboxSampler, input.direction).rgb;

    return vec4<f32>(color, 1.0);
}
`;class O{pipeline;uniformBuffer;uniformBindGroup;depthTexture;cubeVertexBuffer;async init(){const e=o.device.createPipelineLayout({bindGroupLayouts:[o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}}]}),o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{viewDimension:"cube"}}]})]});this.pipeline=o.device.createRenderPipeline({layout:e,vertex:{module:o.device.createShaderModule({code:I}),entryPoint:"main",buffers:[{arrayStride:16,attributes:[{shaderLocation:0,offset:0,format:"float32x4"}]}]},fragment:{module:o.device.createShaderModule({code:D}),entryPoint:"main",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"none"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}});const t=new Float32Array([1,-1,1,1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,-1,1,1,1,-1,-1,1,1,1,-1,1,1,1,1,1,1,-1,-1,1,-1,1,1,1,1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,1,1,-1,1,-1,-1,1,1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,-1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,-1,1,1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,1,1,-1,1,1,-1,-1,1,-1,1,-1,1]);this.cubeVertexBuffer=o.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.cubeVertexBuffer.getMappedRange()).set(t),this.cubeVertexBuffer.unmap(),this.uniformBuffer=o.device.createBuffer({size:128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.uniformBindGroup=o.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),this.depthTexture=o.device.createTexture({size:[800,600],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT})}async draw(e,t,r,i){o.device.queue.writeBuffer(this.uniformBuffer,0,new Float32Array(r)),o.device.queue.writeBuffer(this.uniformBuffer,64,new Float32Array(i));const n=e.beginRenderPass({colorAttachments:[{view:o.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.depthTexture.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});n.setPipeline(this.pipeline),n.setBindGroup(0,this.uniformBindGroup),n.setBindGroup(1,t),n.setVertexBuffer(0,this.cubeVertexBuffer),n.draw(36),n.end()}}const A=new O;class j{fov=70;cframe=new S;getViewMatrix(){return this.cframe.invert().toMatrix()}getProjection(e){return b.perspective(this.fov*Math.PI/180,e,.1,100)}}async function $(u){const e=await Promise.all(u.map(async t=>await q(t)));return H(e)}async function q(u){const t=await(await fetch(u)).blob();return await createImageBitmap(t,{colorSpaceConversion:"none"})}function H(u){const e=u[0],t=o.device.createTexture({format:"rgba8unorm",size:[e.width,e.height,6],usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT,dimension:"2d"});return X(t,u),t}function X(u,e){e.forEach((t,r)=>{o.device.queue.copyExternalImageToTexture({source:t},{texture:u,origin:[0,0,r]},{width:t.width,height:t.height})})}let Y=performance.now();class W{canvas;device;context;format;fragmentBuffer;fragmentBindGroup;skyboxTexture;skyboxSampler;skyboxBindGroup;pbrSkyboxBindGroup;camera;meshes;constructor(){this.canvas=document.getElementById("GLCanvas"),this.meshes=new Array,this.camera=new j;const e=window.devicePixelRatio||1;this.canvas.width=this.canvas.clientWidth*e,this.canvas.height=this.canvas.clientHeight*e}async init(){const e=await navigator.gpu?.requestAdapter();if(!e)throw new Error("Browser does not support WebGPU");if(this.device=await e?.requestDevice(),!this.device)throw new Error("Browser does not support WebGPU");if(this.context=this.canvas.getContext("webgpu"),!this.context)throw new Error("Failed to get canvas context");this.format=navigator.gpu.getPreferredCanvasFormat(),this.context.configure({device:this.device,format:this.format}),await E.init(),await A.init(),this.skyboxSampler=this.device.createSampler(),this.skyboxTexture=await $(["texture/skybox/pos-x.jpg","texture/skybox/neg-x.jpg","texture/skybox/pos-y.jpg","texture/skybox/neg-y.jpg","texture/skybox/pos-z.jpg","texture/skybox/neg-z.jpg"]),this.skyboxBindGroup=this.device.createBindGroup({layout:A.pipeline.getBindGroupLayout(1),entries:[{binding:0,resource:this.skyboxSampler},{binding:1,resource:this.skyboxTexture.createView({dimension:"cube"})}]}),this.pbrSkyboxBindGroup=this.device.createBindGroup({layout:E.pipeline.getBindGroupLayout(3),entries:[{binding:0,resource:this.skyboxSampler},{binding:1,resource:this.skyboxTexture.createView({dimension:"cube"})}]}),this.fragmentBuffer=o.device.createBuffer({size:Float32Array.BYTES_PER_ELEMENT*8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.fragmentBindGroup=o.device.createBindGroup({layout:E.pipeline.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:this.fragmentBuffer}}]})}async loop(){let e=performance.now()-Y,t=Math.cos(e*5e-4)*2,r=1,i=Math.sin(e*5e-4)*2;const n=800/600,a=100*Math.PI/180,p=K(a,n,.1,100),l=J([t,r,i],[0,0,0],[0,1,0]);o.device.queue.writeBuffer(this.fragmentBuffer,0,new Float32Array([t,r,i,8,4,8]));const d=this.device.createCommandEncoder();A.draw(d,this.skyboxBindGroup,l,p);for(let f of this.meshes)E.draw(d,f,l,p,this.fragmentBindGroup,this.pbrSkyboxBindGroup);o.device.queue.submit([d.finish()]),requestAnimationFrame(()=>this.loop())}}function K(u,e,t,r){const i=1/Math.tan(u/2),n=1/(t-r);return new Float32Array([i/e,0,0,0,0,i,0,0,0,0,(r+t)*n,-1,0,0,2*r*t*n,0])}function J(u,e,t){const[r,i,n]=u,[a,c,s]=e,[p,l,d]=t;let f=r-a,m=i-c,g=n-s;const w=Math.hypot(f,m,g);f/=w,m/=w,g/=w;let x=l*g-d*m,y=d*f-p*g,v=p*m-l*f;const B=Math.hypot(x,y,v);x/=B,y/=B,v/=B;const G=m*v-g*y,T=g*x-f*v,z=f*y-m*x;return new Float32Array([x,G,f,0,y,T,m,0,v,z,g,0,-(x*r+y*i+v*n),-(G*r+T*i+z*n),-(f*r+m*i+g*n),1])}let o=new W;class Q extends L{constructor(e=1,t=32,r=32){let i=new Float32Array((t+1)*(r+1)*11),n=new Int16Array(t*r*6),a=0,c=0;for(let s=0;s<=t;s++){const p=s/t,l=p*Math.PI,d=Math.sin(l),f=Math.cos(l);for(let m=0;m<=r;m++){const g=m/r,w=g*Math.PI*2,x=Math.cos(w),y=Math.sin(w),v=e*d*x,B=e*f,G=e*d*y,T=v/e,z=B/e,N=G/e;let P=-y,U=0,F=x;const _=Math.sqrt(P*P+U*U+F*F);P/=_,U/=_,F/=_,i.set([v,B,G,T,z,N,g,1-p,P,U,F],a),a+=11}}for(let s=0;s<t;s++)for(let p=0;p<r;p++){const l=s*(r+1)+p,d=l+r+1;n.set([l,d,l+1],c),c+=3,n.set([d,d+1,l+1],c),c+=3}super(i,n)}}await o.init();let M=new Q;M.cframe.add(new h(0,0,-2));o.meshes.push(M);await o.loop();
