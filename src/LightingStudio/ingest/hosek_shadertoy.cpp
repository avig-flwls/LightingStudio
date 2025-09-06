// Implementation of 2012 Hosek-Wilkie skylight model

// Ground albedo and turbidity are baked into the lookup tables
#define ALBEDO 1
#define TURBIDITY 3

#define M_PI 3.1415926535897932384626433832795
#define CIE_X 0
#define CIE_Y 1
#define CIE_Z 2

float sample_coeff(int channel, int albedo, int turbidity, int quintic_coeff, int coeff) {
    // int index = 540 * albedo + 54 * turbidity + 9 * quintic_coeff + coeff;
    int index =  9 * quintic_coeff + coeff;
	if (channel == CIE_X) return kHosekCoeffsX[index];
	if (channel == CIE_Y) return kHosekCoeffsY[index];
    if (channel == CIE_Z) return kHosekCoeffsZ[index];
}

float sample_radiance(int channel, int albedo, int turbidity, int quintic_coeff) {
    //int index = 60 * albedo + 6 * turbidity + quintic_coeff;
    int index = quintic_coeff;
	if (channel == CIE_X) return kHosekRadX[index];
	if (channel == CIE_Y) return kHosekRadY[index];
	if (channel == CIE_Z) return kHosekRadZ[index];
}

float eval_quintic_bezier(in float[6] control_points, float t) {
	float t2 = t * t;
	float t3 = t2 * t;
	float t4 = t3 * t;
	float t5 = t4 * t;
	
	float t_inv = 1.0 - t;
	float t_inv2 = t_inv * t_inv;
	float t_inv3 = t_inv2 * t_inv;
	float t_inv4 = t_inv3 * t_inv;
	float t_inv5 = t_inv4 * t_inv;
		
	return (
		control_points[0] *             t_inv5 +
		control_points[1] *  5.0 * t  * t_inv4 +
		control_points[2] * 10.0 * t2 * t_inv3 +
		control_points[3] * 10.0 * t3 * t_inv2 +
		control_points[4] *  5.0 * t4 * t_inv  +
		control_points[5] *        t5
	);
}

float transform_sun_zenith(float sun_zenith) {
	float elevation = M_PI / 2.0 - sun_zenith;
		return pow(elevation / (M_PI / 2.0), 0.333333);
}

void get_control_points(int channel, int albedo, int turbidity, int coeff, out float[6] control_points) {
	for (int i = 0; i < 6; ++i) control_points[i] = sample_coeff(channel, albedo, turbidity, i, coeff);
}

void get_control_points_radiance(int channel, int albedo, int turbidity, out float[6] control_points) {
	for (int i = 0; i < 6; ++i) control_points[i] = sample_radiance(channel, albedo, turbidity, i);
}

void get_coeffs(int channel, int albedo, int turbidity, float sun_zenith, out float[9] coeffs) {
	float t = transform_sun_zenith(sun_zenith);
	for (int i = 0; i < 9; ++i) {
		float control_points[6]; 
		get_control_points(channel, albedo, turbidity, i, control_points);
		coeffs[i] = eval_quintic_bezier(control_points, t);
	}
}

vec3 mean_spectral_radiance(int albedo, int turbidity, float sun_zenith) {
	vec3 spectral_radiance;
	for (int i = 0; i < 3; ++i) {
		float control_points[6];
        get_control_points_radiance(i, albedo, turbidity, control_points);
		float t = transform_sun_zenith(sun_zenith);
		spectral_radiance[i] = eval_quintic_bezier(control_points, t);
	}
	return spectral_radiance;
}

float F(float theta, float gamma, in float[9] coeffs) {
	float A = coeffs[0];
	float B = coeffs[1];
	float C = coeffs[2];
	float D = coeffs[3];
	float E = coeffs[4];
	float F = coeffs[5];
	float G = coeffs[6];
	float H = coeffs[8];
	float I = coeffs[7];
	float chi = (1.0 + pow(cos(gamma), 2.0)) / pow(1.0 + H*H - 2.0 * H * cos(gamma), 1.5);
	
	return (
		(1.0 + A * exp(B / (cos(theta) + 0.01))) *
		(C + D * exp(E * gamma) + F * pow(cos(gamma), 2.0) + G * chi + I * sqrt(cos(theta)))
	);
}

vec3 spectral_radiance(float theta, float gamma, int albedo, int turbidity, float sun_zenith) {
	vec3 XYZ;
	for (int i = 0; i < 3; ++i) {
		float coeffs[9];
		get_coeffs(i, albedo, turbidity, sun_zenith, coeffs);
		XYZ[i] = F(theta, gamma, coeffs);
	}
	return XYZ;
}

// Returns angle between two directions defined by zentih and azimuth angles
float angle(float z1, float a1, float z2, float a2) {
	return acos(
		sin(z1) * cos(a1) * sin(z2) * cos(a2) +
		sin(z1) * sin(a1) * sin(z2) * sin(a2) +
		cos(z1) * cos(z2));
}

vec3 sample_sky(float view_zenith, float view_azimuth, float sun_zenith, float sun_azimuth) {
	float gamma = angle(view_zenith, view_azimuth, sun_zenith, sun_azimuth);
	float theta = view_zenith; 
	return spectral_radiance(theta, gamma, ALBEDO, TURBIDITY, sun_zenith) * mean_spectral_radiance(ALBEDO, TURBIDITY, sun_zenith);
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

// Ad-hoc tonemapping, better approach should be used
vec3 tonemap(vec3 color, float exposure) {
	return vec3(2.0) / (vec3(1.0) + exp(-exposure * color)) - vec3(1.0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv    = (fragCoord -0.5 * iResolution.xy) / iResolution.y;
	vec2 mouse = (iMouse.xy -0.5 * iResolution.xy) / iResolution.y;

	float mouse_angle = atan(mouse.x, mouse.y);
	float mouse_distance = clamp(length(mouse) * M_PI, 0.0, M_PI / 2.0 - 0.1);

	float pixel_angle = atan(uv.x,uv.y);
	float pixel_distance =  length(uv) * M_PI;

	if (pixel_distance > M_PI / 2.0) {
		fragColor = vec4(vec3(0.0), 1.0);
		return;
	}

	float sun_zenith = mouse_distance;
	float sun_azimuth = mouse_angle;

	float view_zenith = pixel_distance;
	float view_azimuth = pixel_angle;

	vec3 XYZ = sample_sky(view_zenith, view_azimuth, sun_zenith, sun_azimuth);
	vec3 RGB = XYZ_to_RGB(XYZ);

	vec3 col = tonemap(RGB, 0.1);
	
	fragColor = vec4(col, 1.0);
}