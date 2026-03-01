/**
 * Forward Declarations
 * Include this file to avoid circular dependencies
 */

#ifndef RAYTRACER_FWD_CUH
#define RAYTRACER_FWD_CUH

namespace rt {

// Core types
class Vec3;
using Point3 = Vec3;
using Color = Vec3;
class Ray;
struct Interval;
class AABB;

// Materials
class Material;
enum class MaterialType;

// Geometry
struct HitRecord;
class Hittable;
enum class HittableType;
class Sphere;
class MovingSphere;
class Quad;
class Box;
class Triangle;
class Plane;
struct HittableObject;
class HittableList;

// Acceleration
class BVH;
struct BVHNode;
class BVHBuilder;

// Camera
class Camera;

// Rendering
struct RenderConfig;
enum class ToneMapMode;

// Environment
class Sky;

// Textures
enum class TextureType;
class SolidColor;
class CheckerTexture;
class NoiseTexture;

} // namespace rt

#endif // RAYTRACER_FWD_CUH
