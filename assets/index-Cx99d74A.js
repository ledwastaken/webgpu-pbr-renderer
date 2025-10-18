(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))r(n);new MutationObserver(n=>{for(const i of n)if(i.type==="childList")for(const s of i.addedNodes)s.tagName==="LINK"&&s.rel==="modulepreload"&&r(s)}).observe(document,{childList:!0,subtree:!0});function e(n){const i={};return n.integrity&&(i.integrity=n.integrity),n.referrerPolicy&&(i.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?i.credentials="include":n.crossOrigin==="anonymous"?i.credentials="omit":i.credentials="same-origin",i}function r(n){if(n.ep)return;n.ep=!0;const i=e(n);fetch(n.href,i)}})();class u{x;y;z;constructor(t=0,e=0,r=0){this.x=t,this.y=e,this.z=r}magnitude(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z}unit(){const t=this.magnitude();return t===0?new u(0,0,0):new u(this.x/t,this.y/t,this.z/t)}cross(t){return new u(this.y*t.z-this.z*t.y,this.z*t.x-this.x*t.z,this.x*t.y-this.y*t.x)}negate(){return new u(-this.x,-this.y,-this.z)}add(t){return new u(this.x+t.x,this.y+t.y,this.z+t.z)}subtract(t){return new u(this.x-t.x,this.y-t.y,this.z-t.z)}multiply(t){return new u(this.x*t,this.y*t,this.z*t)}divide(t){if(t===0)throw new Error("Cannot divide by zero");return new u(this.x/t,this.y/t,this.z/t)}addAssign(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this}subtractAssign(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this}multiplyAssign(t){return this.x*=t,this.y*=t,this.z*=t,this}divideAssign(t){if(t===0)throw new Error("Cannot divide by zero");return this.x/=t,this.y/=t,this.z/=t,this}data(){return[this.x,this.y,this.z]}toString(){return`Vector3(${this.x}, ${this.y}, ${this.z})`}}class v{data_;constructor(t){if(t.length!==16)throw new Error("Matrix4 data array must have exactly 16 elements");this.data_=[...t]}static identity(){const t=new Array(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);return new v(t)}static frustum(t,e,r,n,i,s){const c=2*i,o=e-t,l=n-r,f=s-i,a=new Array(c/o,0,0,0,0,-c/l,0,0,(e+t)/o,(n+r)/l,-(s+i)/f,-1,0,0,-c*s/f,0);return new v(a)}static perspective(t,e,r,n){const i=Math.tan(t/2)*r,s=-i,c=i*e,o=-c;return this.frustum(o,c,s,i,r,n)}data(){return[...this.data_]}dataMutable(){return[...this.data_]}get(t,e){if(t<0||t>3||e<0||e>3)throw new Error("Matrix index out of bounds");return this.data_[t*4+e]}set(t,e,r){if(t<0||t>3||e<0||e>3)throw new Error("Matrix index out of bounds");this.data_[t*4+e]=r}toString(){const t=[];for(let e=0;e<4;e++){const r=this.data_.slice(e*4,(e+1)*4);t.push(`[${r.map(n=>n.toFixed(4)).join(", ")}]`)}return`Matrix4(
  ${t.join(`,
  `)}
)`}multiply(t){const e=this.data_,r=t.data_,n=new Array(16);for(let i=0;i<4;i++)for(let s=0;s<4;s++){let c=0;for(let o=0;o<4;o++)c+=e[i*4+o]*r[o*4+s];n[i*4+s]=c}return new v(n)}transpose(){const t=new Array(16);for(let e=0;e<4;e++)for(let r=0;r<4;r++)t[r*4+e]=this.data_[e*4+r];return new v(t)}}class P{r00=1;r01=0;r02=0;r10=0;r11=1;r12=0;r20=0;r21=0;r22=1;pos=new u;toMatrix(){const t=new Array(this.r00,this.r01,this.r02,this.pos.x,this.r10,this.r11,this.r12,this.pos.y,this.r20,this.r21,this.r22,this.pos.z,0,0,0,1);return new v(t)}invert(){let t=new P;return t.r00=this.r00,t.r01=this.r10,t.r02=this.r20,t.r10=this.r01,t.r11=this.r11,t.r12=this.r21,t.r20=this.r02,t.r21=this.r12,t.r22=this.r22,t.pos=new u(-(this.r00*this.pos.x+this.r01*this.pos.y+this.r02*this.pos.z),-(this.r10*this.pos.x+this.r11*this.pos.y+this.r12*this.pos.z),-(this.r20*this.pos.x+this.r21*this.pos.y+this.r22*this.pos.z)),t}add(t){let e=new P;return e.r00=this.r00,e.r01=this.r01,e.r02=this.r02,e.r10=this.r10,e.r11=this.r11,e.r12=this.r12,e.r20=this.r20,e.r21=this.r21,e.r22=this.r22,e.pos=this.pos.add(t),e}multiply(t){return new u(this.r00*t.x+this.r01*t.y+this.r02*t.z+this.pos.x,this.r10*t.x+this.r11*t.y+this.r12*t.z+this.pos.y,this.r20*t.x+this.r21*t.y+this.r22*t.z+this.pos.z)}data(){return[this.r00,this.r01,this.r02,this.r10,this.r11,this.r12,this.r20,this.r21,this.r22,this.pos.x,this.pos.y,this.pos.z]}getPosition(){return new u(this.pos.x,this.pos.y,this.pos.z)}getRight(){return new u(this.r00,this.r10,this.r20)}getUp(){return new u(this.r01,this.r11,this.r21)}getBack(){return new u(this.r02,this.r12,this.r22)}toString(){return`CFrame(pos: ${this.pos.toString()}, rotation: [${this.r00}, ${this.r01}, ${this.r02}, ${this.r10}, ${this.r11}, ${this.r12}, ${this.r20}, ${this.r21}, ${this.r22}])`}}class S{cframe;indicesCount;vertexBuffer;indexBuffer;constructor(t,e){this.cframe=new P,this.indicesCount=e.length,this.vertexBuffer=h.device.createBuffer({size:t.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Float32Array(this.vertexBuffer.getMappedRange()).set(t),this.vertexBuffer.unmap(),this.indexBuffer=h.device.createBuffer({size:e.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST,mappedAtCreation:!0}),new Int16Array(this.indexBuffer.getMappedRange()).set(e),this.indexBuffer.unmap()}}const E=`struct VertexInput {
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
`,U=`struct FragmentInput {
    @location(0) world_pos : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) uv : vec2<f32>,
    @location(3) tangent : vec3<f32>,
    @location(4) bitangent : vec3<f32>,
};

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
    const pi = 3.14159265359;
    let numerator = max(dot(N, X), 0.0);

    let k = alpha / 2.0;
    var denominator = max(dot(N, X), 0.0) * (1.0 - k) + k;
    denominator = max(denominator, 0.00001);

    return numerator / denominator;
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
    let camera_pos = vec3<f32>(1.5, 0.0, 1.5);
    let light_pos = vec3<f32>(1.0, 1.0, 0.1);
    let light_color = vec3<f32>(1.0, 1.0, 0.8);
    let albedo = vec3<f32>(0.8, 0.2, 0.2);
    let roughness = 1.0;
    let emissivity = vec3<f32>(0.01);
    let base_reflectivity = vec3<f32>(0.0);
    let alpha = roughness * roughness;

    let N = normalize(input.normal);
    let V = normalize(camera_pos - input.world_pos);
    let L = normalize(light_pos);
    let H = normalize(V + L);
    let F0 = base_reflectivity;

    let Ks = F(F0, V, H);
    let Kd = vec3<f32>(1.0) - Ks;

    let lambert = albedo / pi;

    let cookTorranceNumerator = D(alpha, N, H) * G(alpha, N, V, L) * F(F0, V, H);
    var cookTorranceDenominator = 4.0 * max(dot(V, N), 0.0) * max(dot(L, N), 0.0);
    cookTorranceDenominator = max(cookTorranceDenominator, 0.00001);
    let cookTorrance = cookTorranceNumerator / cookTorranceDenominator;

    let BRDF = Kd * lambert + cookTorrance;
    let outgoingLight = emissivity + BRDF * light_color * max(dot(L, N), 0.0);

    return vec4<f32>(outgoingLight, 1.0);
}
`;let C=performance.now();class O{pipeline;uniformBuffer;uniformBindGroup;depthTexture;init(){const t=h.device.createPipelineLayout({bindGroupLayouts:[h.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}}]})]});this.pipeline=h.device.createRenderPipeline({layout:t,vertex:{module:h.device.createShaderModule({code:E}),entryPoint:"main",buffers:[{arrayStride:44,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"},{shaderLocation:2,offset:24,format:"float32x2"},{shaderLocation:3,offset:32,format:"float32x3"}]}]},fragment:{module:h.device.createShaderModule({code:U}),entryPoint:"main",targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"front"},depthStencil:{format:"depth24plus",depthWriteEnabled:!0,depthCompare:"less"}}),this.uniformBuffer=h.device.createBuffer({size:256,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.uniformBindGroup=h.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.uniformBuffer}}]}),this.depthTexture=h.device.createTexture({size:[800,600],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT})}draw(t){const e=(performance.now()-C)/1e3,r=800/600,n=Math.PI/2,c=V(n,r,.1,100),o=D([1.5,0,1.5],[0,0,0],[0,1,0]),l=I(e*.3);h.device.queue.writeBuffer(this.uniformBuffer,0,new Float32Array(l)),h.device.queue.writeBuffer(this.uniformBuffer,64,new Float32Array(o)),h.device.queue.writeBuffer(this.uniformBuffer,128,new Float32Array(c));const f=h.device.createCommandEncoder(),a=f.beginRenderPass({colorAttachments:[{view:h.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this.depthTexture.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}});a.setPipeline(this.pipeline),a.setVertexBuffer(0,t.vertexBuffer),a.setIndexBuffer(t.indexBuffer,"uint16"),a.setBindGroup(0,this.uniformBindGroup),a.drawIndexed(t.indicesCount),a.end(),h.device.queue.submit([f.finish()])}}let G=new O;function V(d,t,e,r){const n=1/Math.tan(d/2),i=1/(e-r);return new Float32Array([n/t,0,0,0,0,n,0,0,0,0,(r+e)*i,-1,0,0,2*r*e*i,0])}function D(d,t,e){const[r,n,i]=d,[s,c,o]=t,[l,f,a]=e;let x=r-s,p=n-c,m=i-o;const z=Math.hypot(x,p,m);x/=z,p/=z,m/=z;let y=f*m-a*p,w=a*x-l*m,g=l*p-f*x;const b=Math.hypot(y,w,g);y/=b,w/=b,g/=b;const B=p*g-m*w,F=m*y-x*g,A=x*w-p*y;return new Float32Array([y,B,x,0,w,F,p,0,g,A,m,0,-(y*r+w*n+g*i),-(B*r+F*n+A*i),-(x*r+p*n+m*i),1])}function I(d){const t=Math.cos(d),e=Math.sin(d);return new Float32Array([t,0,e,0,0,1,0,0,-e,0,t,0,0,0,0,1])}class R{fov=70;cframe=new P;getViewMatrix(){return this.cframe.invert().toMatrix()}getProjection(t){return v.perspective(this.fov*Math.PI/180,t,.1,100)}}class ${canvas;device;context;format;camera;meshes;constructor(){this.canvas=document.getElementById("GLCanvas"),this.meshes=new Array,this.camera=new R;const t=window.devicePixelRatio||1;this.canvas.width=this.canvas.clientWidth*t,this.canvas.height=this.canvas.clientHeight*t}async init(){const t=await navigator.gpu?.requestAdapter();if(!t)throw new Error("Browser does not support WebGPU");if(this.device=await t?.requestDevice(),!this.device)throw new Error("Browser does not support WebGPU");if(this.context=this.canvas.getContext("webgpu"),!this.context)throw new Error("Failed to get canvas context");this.format=navigator.gpu.getPreferredCanvasFormat(),this.context.configure({device:this.device,format:this.format}),G.init()}async loop(){for(let t of this.meshes)G.draw(t);requestAnimationFrame(()=>this.loop())}}let h=new $;class k extends S{constructor(t=1,e=32,r=32){let n=new Float32Array((e+1)*(r+1)*11),i=new Int16Array(e*r*6),s=0,c=0;for(let o=0;o<=e;o++){const l=o/e,f=l*Math.PI,a=Math.sin(f),x=Math.cos(f);for(let p=0;p<=r;p++){const m=p/r,z=m*Math.PI*2,y=Math.cos(z),w=Math.sin(z),g=t*a*y,b=t*x,B=t*a*w,F=g/t,A=b/t,L=B/t,_=-a*w,N=0,T=a*y;n.set([g,b,B,F,A,L,m,1-l,_,N,T],s),s+=11}}for(let o=0;o<e;o++)for(let l=0;l<r;l++){const f=o*(r+1)+l,a=f+r+1;i.set([f,a,f+1],c),c+=3,i.set([a,a+1,f+1],c),c+=3}super(n,i)}}await h.init();let M=new k;M.cframe.add(new u(0,0,-2));h.meshes.push(M);await h.loop();
