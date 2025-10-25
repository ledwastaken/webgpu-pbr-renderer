(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const i of document.querySelectorAll('link[rel="modulepreload"]'))r(i);new MutationObserver(i=>{for(const n of i)if(n.type==="childList")for(const u of n.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&r(u)}).observe(document,{childList:!0,subtree:!0});function t(i){const n={};return i.integrity&&(n.integrity=i.integrity),i.referrerPolicy&&(n.referrerPolicy=i.referrerPolicy),i.crossOrigin==="use-credentials"?n.credentials="include":i.crossOrigin==="anonymous"?n.credentials="omit":n.credentials="same-origin",n}function r(i){if(i.ep)return;i.ep=!0;const n=t(i);fetch(i.href,n)}})();class f{x;y;z;constructor(e=0,t=0,r=0){this.x=e,this.y=t,this.z=r}magnitude(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}dot(e){return this.x*e.x+this.y*e.y+this.z*e.z}unit(){const e=this.magnitude();return e===0?new f(0,0,0):new f(this.x/e,this.y/e,this.z/e)}cross(e){return new f(this.y*e.z-this.z*e.y,this.z*e.x-this.x*e.z,this.x*e.y-this.y*e.x)}negate(){return new f(-this.x,-this.y,-this.z)}add(e){return new f(this.x+e.x,this.y+e.y,this.z+e.z)}subtract(e){return new f(this.x-e.x,this.y-e.y,this.z-e.z)}multiply(e){return new f(this.x*e,this.y*e,this.z*e)}divide(e){if(e===0)throw new Error("Cannot divide by zero");return new f(this.x/e,this.y/e,this.z/e)}addAssign(e){return this.x+=e.x,this.y+=e.y,this.z+=e.z,this}subtractAssign(e){return this.x-=e.x,this.y-=e.y,this.z-=e.z,this}multiplyAssign(e){return this.x*=e,this.y*=e,this.z*=e,this}divideAssign(e){if(e===0)throw new Error("Cannot divide by zero");return this.x/=e,this.y/=e,this.z/=e,this}data(){return[this.x,this.y,this.z]}toString(){return`Vector3(${this.x}, ${this.y}, ${this.z})`}}class B{data_;constructor(e){if(e.length!==16)throw new Error("Matrix4 data array must have exactly 16 elements");this.data_=[...e]}static identity(){const e=new Array(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);return new B(e)}static frustum(e,t,r,i,n,u){const c=2*n,s=t-e,h=i-r,a=u-n,l=new Array(c/s,0,0,0,0,-c/h,0,0,(t+e)/s,(i+r)/h,-(u+n)/a,-1,0,0,-c*u/a,0);return new B(l)}static perspective(e,t,r,i){const n=Math.tan(e/2)*r,u=-n,c=n*t,s=-c;return this.frustum(s,c,u,n,r,i)}data(){return[...this.data_]}dataMutable(){return[...this.data_]}get(e,t){if(e<0||e>3||t<0||t>3)throw new Error("Matrix index out of bounds");return this.data_[e*4+t]}set(e,t,r){if(e<0||e>3||t<0||t>3)throw new Error("Matrix index out of bounds");this.data_[e*4+t]=r}toString(){const e=[];for(let t=0;t<4;t++){const r=this.data_.slice(t*4,(t+1)*4);e.push(`[${r.map(i=>i.toFixed(4)).join(", ")}]`)}return`Matrix4(
  ${e.join(`,
  `)}
)`}multiply(e){const t=this.data_,r=e.data_,i=new Array(16);for(let n=0;n<4;n++)for(let u=0;u<4;u++){let c=0;for(let s=0;s<4;s++)c+=t[n*4+s]*r[s*4+u];i[n*4+u]=c}return new B(i)}transpose(){const e=new Array(16);for(let t=0;t<4;t++)for(let r=0;r<4;r++)e[r*4+t]=this.data_[t*4+r];return new B(e)}}class T{r00=1;r01=0;r02=0;r10=0;r11=1;r12=0;r20=0;r21=0;r22=1;pos=new f;toMatrix(){const e=new Array(this.r00,this.r01,this.r02,this.pos.x,this.r10,this.r11,this.r12,this.pos.y,this.r20,this.r21,this.r22,this.pos.z,0,0,0,1);return new B(e)}invert(){let e=new T;return e.r00=this.r00,e.r01=this.r10,e.r02=this.r20,e.r10=this.r01,e.r11=this.r11,e.r12=this.r21,e.r20=this.r02,e.r21=this.r12,e.r22=this.r22,e.pos=new f(-(this.r00*this.pos.x+this.r01*this.pos.y+this.r02*this.pos.z),-(this.r10*this.pos.x+this.r11*this.pos.y+this.r12*this.pos.z),-(this.r20*this.pos.x+this.r21*this.pos.y+this.r22*this.pos.z)),e}add(e){let t=new T;return t.r00=this.r00,t.r01=this.r01,t.r02=this.r02,t.r10=this.r10,t.r11=this.r11,t.r12=this.r12,t.r20=this.r20,t.r21=this.r21,t.r22=this.r22,t.pos=this.pos.add(e),t}multiply(e){return new f(this.r00*e.x+this.r01*e.y+this.r02*e.z+this.pos.x,this.r10*e.x+this.r11*e.y+this.r12*e.z+this.pos.y,this.r20*e.x+this.r21*e.y+this.r22*e.z+this.pos.z)}data(){return[this.r00,this.r01,this.r02,this.r10,this.r11,this.r12,this.r20,this.r21,this.r22,this.pos.x,this.pos.y,this.pos.z]}getPosition(){return new f(this.pos.x,this.pos.y,this.pos.z)}getRight(){return new f(this.r00,this.r10,this.r20)}getUp(){return new f(this.r01,this.r11,this.r21)}getBack(){return new f(this.r02,this.r12,this.r22)}toString(){return`CFrame(pos: ${this.pos.toString()}, rotation: [${this.r00}, ${this.r01}, ${this.r02}, ${this.r10}, ${this.r11}, ${this.r12}, ${this.r20}, ${this.r21}, ${this.r22}])`}}class F{cframe;indicesCount;vertexBuffer;indexBuffer;constructor(e,t){this.cframe=new T,this.indicesCount=t.length,this.vertexBuffer=o.device.createBuffer({size:e.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.vertexBuffer.getMappedRange()).set(e),this.vertexBuffer.unmap(),this.indexBuffer=o.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Int16Array(this.indexBuffer.getMappedRange()).set(t),this.indexBuffer.unmap()}}const N=`struct VertexInput {
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
`,R=`struct FragmentInput {
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
@group(2) @binding(4) var normalSampler: sampler;
@group(2) @binding(5) var normalData: texture_2d<f32>;

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

    let TBN = mat3x3<f32>(
        input.tangent,
        input.bitangent,
        input.normal,
    );
    let normalMap = 2 * textureSample(normalData, normalSampler, input.uv).rgb - vec3<f32>(1.0);

    let N = normalize(TBN * normalMap);//normalize(input.normal);
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
`;class k{pipeline;uniformBuffer;uniformBindGroup;depthTexture;albedoSampler;albedoTexture;roughnessSampler;roughnessTexture;normalSampler;normalTexture;textureBindGroup;async init(){const e=o.device.createPipelineLayout({bindGroupLayouts:[o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}}]}),o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}}]}),o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{}},{binding:4,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:5,visibility:GPUShaderStage.FRAGMENT,texture:{}}]}),o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{viewDimension:"cube"}}]})]});this.pipeline=o.device.createRenderPipeline({layout:e,vertex:{module:o.device.createShaderModule({code:N}),entryPoint:"main",buffers:[{arrayStride:44,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x2"},{shaderLocation:3,offset:32,format:"float32x3"}]}]},fragment:{module:o.device.createShaderModule({code:R}),entryPoint:"main",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"front"},multisample:{count:4},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.uniformBuffer=o.device.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.uniformBindGroup=o.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),[this.roughnessTexture,this.roughnessSampler]=await this.loadTexture("texture/Metal046B_2K-JPG_Roughness.jpg"),[this.albedoTexture,this.albedoSampler]=await this.loadTexture("texture/Metal046B_2K-JPG_Color.jpg"),[this.normalTexture,this.normalSampler]=await this.loadTexture("texture/Metal046B_2K-JPG_NormalGL.jpg"),this.textureBindGroup=o.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(2),entries:[{binding:0,resource:this.albedoSampler},{binding:1,resource:this.albedoTexture.createView()},{binding:2,resource:this.roughnessSampler},{binding:3,resource:this.roughnessTexture.createView()},{binding:4,resource:this.normalSampler},{binding:5,resource:this.normalTexture.createView()}]}),this.depthTexture=o.device.createTexture({size:[800,600],sampleCount:4,format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT})}async loadTexture(e){const t=document.createElement("img");t.src=e,await t.decode();let r=await createImageBitmap(t),i=o.device.createTexture({size:[r.width,r.height,1],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}),n=o.device.createSampler({magFilter:"linear",minFilter:"linear"});return o.device.queue.copyExternalImageToTexture({source:r},{texture:i},[r.width,r.height]),[i,n]}draw(e,t,r,i,n,u){const c=V(0);o.device.queue.writeBuffer(this.uniformBuffer,0,new Float32Array(c)),o.device.queue.writeBuffer(this.uniformBuffer,64,new Float32Array(r)),o.device.queue.writeBuffer(this.uniformBuffer,128,new Float32Array(i));const s=e.beginRenderPass({colorAttachments:[{view:o.msaaColorTexture.createView(),resolveTarget:o.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"load",storeOp:"store"}],depthStencilAttachment:{view:this.depthTexture.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});s.setPipeline(this.pipeline),s.setVertexBuffer(0,t.vertexBuffer),s.setIndexBuffer(t.indexBuffer,"uint16"),s.setBindGroup(0,this.uniformBindGroup),s.setBindGroup(1,n),s.setBindGroup(2,this.textureBindGroup),s.setBindGroup(3,u),s.drawIndexed(t.indicesCount),s.end()}}function V(g){const e=Math.cos(g),t=Math.sin(g);return new Float32Array([e,0,t,0,0,1,0,0,-t,0,e,0,0,0,0,1])}const U=new k,D=`struct VertexInput {
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
`,I=`struct FragmentInput {
    @location(0) direction : vec3<f32>,
};

@group(1) @binding(0) var skyboxSampler: sampler;
@group(1) @binding(1) var skyboxData: texture_cube<f32>;

@fragment
fn main(input: FragmentInput) -> @location(0) vec4<f32> {
    let color = textureSample(skyboxData, skyboxSampler, input.direction).rgb;

    return vec4<f32>(color, 1.0);
}
`;class O{pipeline;uniformBuffer;uniformBindGroup;depthTexture;cubeVertexBuffer;async init(){const e=o.device.createPipelineLayout({bindGroupLayouts:[o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}}]}),o.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.FRAGMENT,sampler:{}},{binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{viewDimension:"cube"}}]})]});this.pipeline=o.device.createRenderPipeline({layout:e,vertex:{module:o.device.createShaderModule({code:D}),entryPoint:"main",buffers:[{arrayStride:16,attributes:[{shaderLocation:0,offset:0,format:"float32x4"}]}]},fragment:{module:o.device.createShaderModule({code:I}),entryPoint:"main",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"none"},multisample:{count:4},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}});const t=new Float32Array([1,-1,1,1,-1,-1,1,1,-1,-1,-1,1,1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,-1,1,1,1,-1,-1,1,1,1,-1,1,1,1,1,1,1,-1,-1,1,-1,1,1,1,1,1,1,1,1,1,-1,1,-1,1,-1,1,-1,1,1,1,1,1,-1,1,-1,-1,1,1,-1,1,1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,-1,1,1,1,-1,-1,1,1,-1,-1,1,1,1,-1,1,1,1,1,1,1,1,-1,-1,1,-1,-1,-1,1,-1,1,-1,1,1,1,-1,1,1,-1,-1,1,-1,1,-1,1]);this.cubeVertexBuffer=o.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.cubeVertexBuffer.getMappedRange()).set(t),this.cubeVertexBuffer.unmap(),this.uniformBuffer=o.device.createBuffer({size:128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.uniformBindGroup=o.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),this.depthTexture=o.device.createTexture({size:[800,600],sampleCount:4,format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT})}async draw(e,t,r,i){o.device.queue.writeBuffer(this.uniformBuffer,0,new Float32Array(r)),o.device.queue.writeBuffer(this.uniformBuffer,64,new Float32Array(i));const n=e.beginRenderPass({colorAttachments:[{view:o.msaaColorTexture.createView(),resolveTarget:o.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.depthTexture.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});n.setPipeline(this.pipeline),n.setBindGroup(0,this.uniformBindGroup),n.setBindGroup(1,t),n.setVertexBuffer(0,this.cubeVertexBuffer),n.draw(36),n.end()}}const L=new O;class ${fov=70;cframe=new T;getViewMatrix(){return this.cframe.invert().toMatrix()}getProjection(e){return B.perspective(this.fov*Math.PI/180,e,.1,100)}}let j=performance.now();class q{canvas;device;context;format;msaaColorTexture;fragmentBuffer;fragmentBindGroup;skyboxTexture;skyboxSampler;skyboxBindGroup;pbrSkyboxBindGroup;camera;meshes;constructor(){this.canvas=document.getElementById("GLCanvas"),this.meshes=new Array,this.camera=new $;const e=window.devicePixelRatio||1;this.canvas.width=this.canvas.clientWidth*e,this.canvas.height=this.canvas.clientHeight*e}async init(){const e=await navigator.gpu?.requestAdapter();if(!e)throw new Error("Browser does not support WebGPU");if(this.device=await e?.requestDevice(),!this.device)throw new Error("Browser does not support WebGPU");if(this.context=this.canvas.getContext("webgpu"),!this.context)throw new Error("Failed to get canvas context");this.format=navigator.gpu.getPreferredCanvasFormat(),this.context.configure({device:this.device,format:this.format});const t=navigator.gpu.getPreferredCanvasFormat();this.msaaColorTexture=this.device.createTexture({size:[this.canvas.width,this.canvas.height],sampleCount:4,format:t,usage:GPUTextureUsage.RENDER_ATTACHMENT}),await U.init(),await L.init();const r=await Promise.all(["texture/skybox/pos-x.jpg","texture/skybox/neg-x.jpg","texture/skybox/pos-y.jpg","texture/skybox/neg-y.jpg","texture/skybox/pos-z.jpg","texture/skybox/neg-z.jpg"].map(async a=>{const d=await(await fetch(a)).blob();return await createImageBitmap(d,{colorSpaceConversion:"none"})})),i=r[0].width,n=r[0].height,u=Math.floor(Math.log2(Math.max(i,n)))+1;this.skyboxTexture=this.device.createTexture({size:[i,n,6],mipLevelCount:u,format:"rgba8unorm",dimension:"2d",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});for(let a=0;a<6;a++)this.device.queue.copyExternalImageToTexture({source:r[a],flipY:!1},{texture:this.skyboxTexture,mipLevel:0,origin:[0,0,a]},[i,n,1]);this.skyboxSampler=this.device.createSampler();const c=this.device.createShaderModule({code:`
                @group(0) @binding(0) var srcTexture : texture_2d<f32>;
                @group(0) @binding(1) var dstTexture : texture_storage_2d<rgba8unorm, write>;

                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) global_id : vec3u) {
                    let dstSize = textureDimensions(dstTexture);
                    if (global_id.x >= dstSize.x || global_id.y >= dstSize.y) {
                        return;
                    }

                    let srcCoord = vec2i(global_id.xy * 2u);

                    var color = textureLoad(srcTexture, srcCoord, 0);
                    color += textureLoad(srcTexture, srcCoord + vec2i(1, 0), 0);
                    color += textureLoad(srcTexture, srcCoord + vec2i(0, 1), 0);
                    color += textureLoad(srcTexture, srcCoord + vec2i(1, 1), 0);
                    color /= 4.0;

                    textureStore(dstTexture, vec2i(global_id.xy), color);
                }
            `}),s=this.device.createComputePipeline({layout:"auto",compute:{module:c,entryPoint:"main"}}),h=this.device.createCommandEncoder();for(let a=0;a<6;a++){let l=i,d=n;for(let p=1;p<u;p++){l=Math.max(1,Math.floor(l/2)),d=Math.max(1,Math.floor(d/2));const x=this.skyboxTexture.createView({dimension:"2d",baseMipLevel:p-1,mipLevelCount:1,baseArrayLayer:a,arrayLayerCount:1}),b=this.skyboxTexture.createView({dimension:"2d",baseMipLevel:p,mipLevelCount:1,baseArrayLayer:a,arrayLayerCount:1}),y=this.device.createBindGroup({layout:s.getBindGroupLayout(0),entries:[{binding:0,resource:x},{binding:1,resource:b}]}),m=h.beginComputePass();m.setPipeline(s),m.setBindGroup(0,y);const v=Math.ceil(l/8),w=Math.ceil(d/8);m.dispatchWorkgroups(v,w,1),m.end()}}this.device.queue.submit([h.finish()]),r.forEach(a=>a.close()),this.skyboxBindGroup=this.device.createBindGroup({layout:L.pipeline.getBindGroupLayout(1),entries:[{binding:0,resource:this.skyboxSampler},{binding:1,resource:this.skyboxTexture.createView({dimension:"cube"})}]}),this.pbrSkyboxBindGroup=this.device.createBindGroup({layout:U.pipeline.getBindGroupLayout(3),entries:[{binding:0,resource:this.skyboxSampler},{binding:1,resource:this.skyboxTexture.createView({dimension:"cube"})}]}),this.fragmentBuffer=this.device.createBuffer({size:Float32Array.BYTES_PER_ELEMENT*8,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.fragmentBindGroup=this.device.createBindGroup({layout:U.pipeline.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:this.fragmentBuffer}}]})}async loop(){let e=performance.now()-j,t=Math.cos(e*3e-4)*2,r=1,i=Math.sin(e*3e-4)*2;const n=800/600,u=100*Math.PI/180,h=H(u,n,.1,100),a=X([t,r,i],[0,0,0],[0,1,0]);this.device.queue.writeBuffer(this.fragmentBuffer,0,new Float32Array([t,r,i,8,4,8]));const l=this.device.createCommandEncoder();L.draw(l,this.skyboxBindGroup,a,h);for(let d of this.meshes)U.draw(l,d,a,h,this.fragmentBindGroup,this.pbrSkyboxBindGroup);this.device.queue.submit([l.finish()]),requestAnimationFrame(()=>this.loop())}}function H(g,e,t,r){const i=1/Math.tan(g/2),n=1/(t-r);return new Float32Array([i/e,0,0,0,0,i,0,0,0,0,(r+t)*n,-1,0,0,2*r*t*n,0])}function X(g,e,t){const[r,i,n]=g,[u,c,s]=e,[h,a,l]=t;let d=r-u,p=i-c,x=n-s;const b=Math.hypot(d,p,x);d/=b,p/=b,x/=b;let y=a*x-l*p,m=l*d-h*x,v=h*p-a*d;const w=Math.hypot(y,m,v);y/=w,m/=w,v/=w;const G=p*v-x*m,S=x*y-d*v,P=d*m-p*y;return new Float32Array([y,G,d,0,m,S,p,0,v,P,x,0,-(y*r+m*i+v*n),-(G*r+S*i+P*n),-(d*r+p*i+x*n),1])}const o=new q;class Y extends F{constructor(e=1,t=32,r=32){let i=new Float32Array((t+1)*(r+1)*11),n=new Int16Array(t*r*6),u=0,c=0;for(let s=0;s<=t;s++){const h=s/t,a=h*Math.PI,l=Math.sin(a),d=Math.cos(a);for(let p=0;p<=r;p++){const x=p/r,b=x*Math.PI*2,y=Math.cos(b),m=Math.sin(b),v=e*l*y,w=e*d,G=e*l*m,S=v/e,P=w/e,A=G/e;let z=-m,M=0,_=y;const E=Math.sqrt(z*z+M*M+_*_);z/=E,M/=E,_/=E,i.set([v,w,G,S,P,A,x,1-h,z,M,_],u),u+=11}}for(let s=0;s<t;s++)for(let h=0;h<r;h++){const a=s*(r+1)+h,l=a+r+1;n.set([a,l,a+1],c),c+=3,n.set([l,l+1,a+1],c),c+=3}super(i,n)}}await o.init();let C=new Y(1,64,64);C.cframe.add(new f(0,0,-2));o.meshes.push(C);await o.loop();
