import { Mesh } from "../scene/mesh";

export class Cube extends Mesh {
  public constructor(size = 1) {
    const unit = size;
    
    // Each vertex has: position (3), normal (3), uv (2), tangent (3) = 11 values
    const vertices = new Float32Array([
      // Right face (+X)
      unit, -unit, -unit,  1, 0, 0,  0.375, 0.5,   0, 0, 1,
      unit, -unit, unit,   1, 0, 0,  0.625, 0.5,   0, 0, 1,
      unit, unit, unit,    1, 0, 0,  0.625, 0.75,  0, 0, 1,
      unit, unit, -unit,   1, 0, 0,  0.375, 0.75,  0, 0, 1,
      
      // Front face (+Z)
      -unit, -unit, unit,  0, 0, 1,  0.875, 0.5,   1, 0, 0,
      -unit, unit, unit,   0, 0, 1,  0.875, 0.75,  1, 0, 0,
      unit, unit, unit,    0, 0, 1,  0.625, 0.75,  1, 0, 0,
      unit, -unit, unit,   0, 0, 1,  0.625, 0.5,   1, 0, 0,
      
      // Left face (-X)
      -unit, -unit, unit,   -1, 0, 0,  0.625, 0.5,   0, 0, -1,
      -unit, -unit, -unit,  -1, 0, 0,  0.375, 0.5,   0, 0, -1,
      -unit, unit, -unit,   -1, 0, 0,  0.375, 0.75,  0, 0, -1,
      -unit, unit, unit,    -1, 0, 0,  0.625, 0.75,  0, 0, -1,
      
      // Back face (-Z)
      unit, -unit, -unit,  0, 0, -1,  0.625, 0.0,   -1, 0, 0,
      unit, unit, -unit,   0, 0, -1,  0.625, 0.25,  -1, 0, 0,
      -unit, unit, -unit,  0, 0, -1,  0.875, 0.25,  -1, 0, 0,
      -unit, -unit, -unit, 0, 0, -1,  0.875, 0.0,   -1, 0, 0,
      
      // Top face (+Y)
      -unit, unit, unit,   0, 1, 0,  0.375, 0.75,  1, 0, 0,
      -unit, unit, -unit,  0, 1, 0,  0.375, 1.0,   1, 0, 0,
      unit, unit, -unit,   0, 1, 0,  0.625, 1.0,   1, 0, 0,
      unit, unit, unit,    0, 1, 0,  0.625, 0.75,  1, 0, 0,
      
      // Bottom face (-Y)
      -unit, -unit, -unit, 0, -1, 0,  0.375, 0.25,  1, 0, 0,
      -unit, -unit, unit,  0, -1, 0,  0.375, 0.5,   1, 0, 0,
      unit, -unit, unit,   0, -1, 0,  0.625, 0.5,   1, 0, 0,
      unit, -unit, -unit,  0, -1, 0,  0.625, 0.25,  1, 0, 0,
    ]);
    
    const indices = new Int16Array([
      0, 1, 2,    2, 3, 0,    // Right
      4, 5, 6,    6, 7, 4,    // Front
      8, 9, 10,   10, 11, 8,  // Left
      12, 13, 14, 14, 15, 12, // Back
      16, 17, 18, 18, 19, 16, // Top
      20, 21, 22, 22, 23, 20, // Bottom
    ]);
    
    super(vertices, indices);
  }
}
