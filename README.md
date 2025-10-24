# WebGPU PBR Renderer

A physically based rendering (PBR) engine built with TypeScript and WebGPU.
This project makes use of advanced rendering techniques such as mipmapping and multisampling to enhance image quality and reduce aliasing artifacts.

<video autoplay muted loop playsinline src="https://ledwastaken.github.io/webgpu-pbr-renderer/sample.mp4"></video>

You can view the live demo of the web application here: https://ledwastaken.github.io/webgpu-pbr-renderer/

## Getting Started

### Development Environment

This project uses Nix to provide a reproducible development environment.
If you're on NixOS (or have Nix installed), simply run:
```bash
nix develop
```
This will open a shell with all required dependencies (Node.js, TypeScript, etc.).

Install dependencies and start the dev server:
```bash
npm install
npm run dev
```
Then open the URL printed in your terminal (usually http://localhost:5173).

### Build instructions

To bundle the project for production:
```bash
npm run build
```
The output will be available in the dist/ directory.
