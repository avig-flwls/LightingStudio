// Implementation of 1999 preetham skylight model

#define T 2.3
#define M_PI 3.1415926535897932384626433832795

// Radiance distribution parameters
float A_x = -0.0193 * T - 0.2592;
float B_x = -0.0665 * T + 0.0008;
float C_x = -0.0004 * T + 0.2125;
float D_x = -0.0641 * T - 0.8989;
float E_x = -0.0033 * T + 0.0452;

float A_y = -0.0167 * T - 0.2608;
float B_y = -0.0950 * T + 0.0092;
float C_y = -0.0079 * T + 0.2102;
float D_y = -0.0441 * T - 1.6537;
float E_y = -0.0109 * T + 0.0529;

float A_Y =  0.1787 * T - 1.4630;
float B_Y = -0.3554 * T + 0.4275;
float C_Y = -0.0227 * T + 5.3251;
float D_Y =  0.1206 * T - 2.5771;
float E_Y = -0.0670 * T + 0.3703;

// Matrices for chromaticity calculations
mat4x3 x_chromaticity = mat4x3(
    0.0017, -0.0290, 0.1169,
   -0.0037,  0.0638,-0.2120,
    0.0021, -0.0320, 0.0605,
    0.0000,  0.0039, 0.2589
);

mat4x3 y_chromaticity = mat4x3(
    0.0028, -0.0421,  0.1535,
   -0.0061,  0.0897, -0.2676,
    0.0032, -0.0415,  0.0667,
    0.0000,  0.0052,  0.2669
);

float angle(float z1, float a1, float z2, float a2) {
	return acos(
        sin(z1) * cos(a1) * sin(z2) * cos(a2) +
        sin(z1) * sin(a1) * sin(z2) * sin(a2) +
        cos(z1) * cos(z2));
}

float zenith_chromaticity(float sun_z, mat4x3 coefficients) {
    vec3 T_vec = vec3(T * T, T, 1);
    vec4 Z_vec = vec4(sun_z*sun_z*sun_z, sun_z*sun_z, sun_z, 1.0);
    return dot(T_vec, coefficients * Z_vec);
}

float zenith_luminance(float sun_z) {
 	float chi = (4.0 / 9.0 - T / 120.0) * (M_PI - 2.0 * sun_z);
    return (4.0453 * T - 4.9710) * tan(chi) - 0.2155 * T + 2.4192;
}

float F(float theta, float gamma, float A, float B, float C, float D, float E) {
	return (1.0 + A * exp(B / cos(theta))) * (1.0 + C * exp(D * gamma) + E * pow(cos(gamma), 2.0));
}

// CIE-xyY to CIE-XYZ
vec3 xyY_to_XYZ(float x, float y, float Y) {
	return vec3(x * Y / y, Y, (1.0 - x - y) * Y / y);
}

// CIE-XYZ to linear RGB
vec3 XYZ_to_RGB(vec3 XYZ) {
    mat3 XYZ_to_linear = mat3(
  		 3.24096994, -0.96924364, 0.55630080,
        -1.53738318,  1.8759675, -0.20397696,
        -0.49861076,  0.04155506, 1.05697151
    );
    return XYZ_to_linear * XYZ;
}

// CIE-xyY to RGB
vec3 xyY_to_RGB(float x, float y, float Y) {
 	vec3 XYZ = xyY_to_XYZ(x, y, Y);
   	vec3 sRGB = XYZ_to_RGB(XYZ);
    return sRGB;
}

// Ad-hoc tonemap
vec3 tonemap(vec3 color, float exposure) {
    return vec3(2.0) / (vec3(1.0) + exp(-exposure * color)) - vec3(1.0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv    = (fragCoord -0.5 * iResolution.xy) / iResolution.y;
    vec2 mouse = (iMouse.xy -0.5 * iResolution.xy) / iResolution.y;
    
    float mouse_angle = atan(mouse.x, mouse.y);
    float mouse_distance = clamp(length(mouse) * M_PI, 0.0, M_PI / 2.0);
    
    float pixel_angle = atan(uv.x,uv.y);
    float pixel_distance =  length(uv) * M_PI;
    
    if (pixel_distance > M_PI / 2.0) {
    	fragColor = vec4(vec3(0.0), 1.0);
        return;
    }
    
    float sun_zenith = mouse_distance;
    float sun_azimuth = mouse_angle;
    
    float zenith = pixel_distance;
    float azimuth = pixel_angle;
    
    float gamma = angle(zenith, azimuth, sun_zenith, sun_azimuth);
    float theta = zenith;
    
    float x_z = zenith_chromaticity(sun_zenith, x_chromaticity);
    float y_z = zenith_chromaticity(sun_zenith, y_chromaticity);
    float Y_z = zenith_luminance(sun_zenith);
    
    float x = x_z * F(theta, gamma, A_x, B_x, C_x, D_x, E_x) / F(0.0, sun_zenith, A_x, B_x, C_x, D_x, E_x);
    float y = y_z * F(theta, gamma, A_y, B_y, C_y, D_y, E_y) / F(0.0, sun_zenith, A_y, B_y, C_y, D_y, E_y);
  	float Y = Y_z * F(theta, gamma, A_Y, B_Y, C_Y, D_Y, E_Y) / F(0.0, sun_zenith, A_Y, B_Y, C_Y, D_Y, E_Y);

    vec3 col = tonemap(xyY_to_RGB(x, y, Y), 0.1);
    
    fragColor = vec4(col, 1.0);
}