/**
 * Material Factory - Convenience Header
 * Includes all material creation functions
 */

#ifndef RAYTRACER_MATERIALS_MATERIAL_FACTORY_CUH
#define RAYTRACER_MATERIALS_MATERIAL_FACTORY_CUH

#include "raytracer/materials/material.cuh"
#include "raytracer/materials/lambertian.cuh"
#include "raytracer/materials/metal.cuh"
#include "raytracer/materials/dielectric.cuh"
#include "raytracer/materials/emissive.cuh"
#include "raytracer/materials/isotropic.cuh"

// All material creation functions are available:
//
// create_lambertian(Color albedo)
// create_lambertian_checker(float scale, Color c1, Color c2)
// create_lambertian_noise(float scale, Color base_color, unsigned seed)
// create_metal(Color albedo, float fuzz)
// create_dielectric(float ior)
// create_emissive(Color emit, float strength)
// create_isotropic(Color albedo)

#endif // RAYTRACER_MATERIALS_MATERIAL_FACTORY_CUH
