var _____WB$wombat$assign$function_____ = function(name) {return (self._wb_wombat && self._wb_wombat.local_init && self._wb_wombat.local_init(name)) || self[name]; };
if (!self.__WB_pmw) { self.__WB_pmw = function(obj) { this.__WB_source = obj; return this; } }
{
  let window = _____WB$wombat$assign$function_____("window");
  let self = _____WB$wombat$assign$function_____("self");
  let document = _____WB$wombat$assign$function_____("document");
  let location = _____WB$wombat$assign$function_____("location");
  let top = _____WB$wombat$assign$function_____("top");
  let parent = _____WB$wombat$assign$function_____("parent");
  let frames = _____WB$wombat$assign$function_____("frames");
  let opener = _____WB$wombat$assign$function_____("opener");

//Some global variables
var demoSHLightingExtractDLight= new Object();
demoSHLightingExtractDLight.gl= null;
demoSHLightingExtractDLight.canvas= null;

// budda and skybox model deleted.

demoSHLightingExtractDLight.skyboxVertexBuffer= null;

demoSHLightingExtractDLight.modelWorldMatrix= null;
demoSHLightingExtractDLight.viewMatrix= null;
demoSHLightingExtractDLight.projectionMatrix= null;




demoSHLightingExtractDLight.onOffEum            = {on:0, off:1, Num:2};

demoSHLightingExtractDLight.shaderSHLightingVS = "attribute vec3 a_position;\n\
attribute vec3 a_normal;\n\
varying vec3 v_position;\n\
varying vec3 v_normal;\n\
uniform mat4 u_worldMat;\n\
uniform mat4 u_worldViewProjMat;\n\
void main() {\n\
  v_normal = (u_worldMat * vec4(a_normal, 0.0)).xyz;\n\
  v_position = (u_worldMat * vec4(a_position, 1.0)).xyz;\n\
  gl_Position = u_worldViewProjMat * vec4(a_position, 1.0);\n\
}\n";

demoSHLightingExtractDLight.shaderSHLightingFS_def = "precision highp float;\n\
varying vec3 v_position;\n\
varying vec3 v_normal;\n\
uniform float u_glossiness;\n\
uniform vec3 u_cameraPosition;\n\
uniform vec3 u_light_SHCoef0_r;\n\
uniform vec3 u_light_SHCoef1_r;\n\
uniform vec3 u_light_SHCoef2_r;\n\
uniform vec3 u_light_SHCoef0_g;\n\
uniform vec3 u_light_SHCoef1_g;\n\
uniform vec3 u_light_SHCoef2_g;\n\
uniform vec3 u_light_SHCoef0_b;\n\
uniform vec3 u_light_SHCoef1_b;\n\
uniform vec3 u_light_SHCoef2_b;\n\
uniform vec3 u_transferFunc_ZHCoef;\n\
uniform vec3 u_lambertBRDF;\n";

demoSHLightingExtractDLight.shaderSHLightingFS_diffuse= new Array(demoSHLightingExtractDLight.onOffEum.Num);
demoSHLightingExtractDLight.shaderSHLightingFS_diffuse[demoSHLightingExtractDLight.onOffEum.on] = "vec3 diffuseLighting(vec3 sh0_r, vec3 sh1_r, vec3 sh2_r, vec3 sh0_g, vec3 sh1_g, vec3 sh2_g, vec3 sh0_b, vec3 sh1_b, vec3 sh2_b, vec3 lambertBrdf, vec3 zhTranFunc, vec3 normal) {\n\
  float sqrt3 = 1.732050808;\n\
  /* rotate ZH coef to world space SH in spherical coordinates (i.e. z-axiz is up axis)*/\n\
  vec3 transferFunc_SHCoef0, transferFunc_SHCoef1, transferFunc_SHCoef2;\n\
  transferFunc_SHCoef0.x = zhTranFunc.x; /* 0-band */\n\
  transferFunc_SHCoef0.y = -zhTranFunc.y * normal.y; /* 1-band */\n\
  transferFunc_SHCoef0.z = zhTranFunc.y * normal.z;\n\
  transferFunc_SHCoef1.x = -zhTranFunc.y * normal.x;\n\
  transferFunc_SHCoef1.y = sqrt3 * zhTranFunc.z * normal.x * normal.y; /* 2-band*/\n\
  transferFunc_SHCoef1.z = -sqrt3 * zhTranFunc.z * normal.y * normal.z;\n\
  transferFunc_SHCoef2.x = 0.5 * zhTranFunc.z * (3.0 * normal.z * normal.z - 1.0);\n\
  transferFunc_SHCoef2.y = -sqrt3 * zhTranFunc.z * normal.x * normal.z;\n\
  transferFunc_SHCoef2.z = 0.5 * sqrt3 * zhTranFunc.z * (normal.x * normal.x - normal.y * normal.y);\n\
  return vec3(\n\
    lambertBrdf.r * (dot(sh0_r, transferFunc_SHCoef0) + dot(sh1_r, transferFunc_SHCoef1) + dot(sh2_r, transferFunc_SHCoef2)),\n\
    lambertBrdf.g * (dot(sh0_g, transferFunc_SHCoef0) + dot(sh1_g, transferFunc_SHCoef1) + dot(sh2_g, transferFunc_SHCoef2)),\n\
    lambertBrdf.b * (dot(sh0_b, transferFunc_SHCoef0) + dot(sh1_b, transferFunc_SHCoef1) + dot(sh2_b, transferFunc_SHCoef2))\n\
  );\n\
}\n";
demoSHLightingExtractDLight.shaderSHLightingFS_diffuse[demoSHLightingExtractDLight.onOffEum.off] = "vec3 diffuseLighting(vec3 sh0_r, vec3 sh1_r, vec3 sh2_r, vec3 sh0_g, vec3 sh1_g, vec3 sh2_g, vec3 sh0_b, vec3 sh1_b, vec3 sh2_b, vec3 lambertBrdf, vec3 zhTranFunc, vec3 normal) {\n\
  return vec3(0, 0, 0);\n\
}\n";

demoSHLightingExtractDLight.shaderSHLightingFS_spec= new Array(demoSHLightingExtractDLight.onOffEum.Num);
demoSHLightingExtractDLight.shaderSHLightingFS_spec[demoSHLightingExtractDLight.onOffEum.on] = "vec3 specLighting(vec3 sh0_r, vec3 sh1_r, vec3 sh2_r, vec3 sh0_g, vec3 sh1_g, vec3 sh2_g, vec3 sh0_b, vec3 sh1_b, vec3 sh2_b, vec3 lambertBrdf, vec3 view, vec3 normal, float glossiness) {\n\
  vec3 lightDirR = normalize(vec3(-sh1_r.x, -sh0_r.y, sh0_r.z));\n\
  vec3 lightDirG = normalize(vec3(-sh1_g.x, -sh0_g.y, sh0_g.z));\n\
  vec3 lightDirB = normalize(vec3(-sh1_b.x, -sh0_b.y, sh0_b.z));\n\
  vec3 lightDir = normalize(0.3*lightDirR + 0.59*lightDirG + 0.11*lightDirB);\n\
  /* project the directional light into SH */\n\
  vec3 sh0_light = vec3(0.282094791, -0.488602511*lightDir.y, 0.488602511*lightDir.z);\n\
  vec3 sh1_light = vec3(-0.488602511*lightDir.x, 1.092548431*lightDir.y*lightDir.x, -1.092548431*lightDir.y*lightDir.z);\n\
  vec3 sh2_light = vec3(0.315391565*(3.0*lightDir.z*lightDir.z-1.0), -1.092548431*lightDir.x*lightDir.z, 0.546274215*(lightDir.x*lightDir.x - lightDir.y*lightDir.y));\n\
  /* normalize the light */\n\
  sh0_light *= 2.956793086;\n\
  sh1_light *= 2.956793086;\n\
  sh2_light *= 2.956793086;\n\
  float denom = dot(sh0_light, sh0_light) + dot(sh1_light, sh1_light) + dot(sh2_light, sh2_light);\n\
  vec3 lightColor = vec3(\n\
    dot(sh0_r, sh0_light) + dot(sh1_r, sh1_light) + dot(sh2_r, sh2_light),\n\
    dot(sh0_g, sh0_light) + dot(sh1_g, sh1_light) + dot(sh2_g, sh2_light),\n\
    dot(sh0_b, sh0_light) + dot(sh1_b, sh1_light) + dot(sh2_b, sh2_light)\n\
  );\n\
  lightColor /= denom;\n\
  vec3 H = normalize(lightDir + view);\n\
  float NdotH = max(dot(normal, H), 0.0);\n\
  float NdotL = max(dot(normal, lightDir), 0.0);\n\
  return 3.141592654*lambertBrdf*lightColor*pow(NdotH, glossiness)*NdotL; /* multiply by PI because we divided PI in lambertBrdf*/\n\
}\n";

demoSHLightingExtractDLight.shaderSHLightingFS_spec[demoSHLightingExtractDLight.onOffEum.off] = "vec3 specLighting(vec3 sh0_r, vec3 sh1_r, vec3 sh2_r, vec3 sh0_g, vec3 sh1_g, vec3 sh2_g, vec3 sh0_b, vec3 sh1_b, vec3 sh2_b, vec3 lambertBrdf, vec3 view, vec3 normal, float glossiness) {\n\
  return vec3(0, 0, 0);\n\
}\n";


demoSHLightingExtractDLight.shaderSHLightingFS_main = "void main() {\n\
  vec3 normal = normalize(v_normal).xzy; /*transform to spherical coordinates with z-axiz as up axis*/\n\
  vec3 view = normalize(u_cameraPosition - v_position).xzy;\n\
  vec3 diffuse = diffuseLighting(u_light_SHCoef0_r, u_light_SHCoef1_r, u_light_SHCoef2_r, u_light_SHCoef0_g, u_light_SHCoef1_g, u_light_SHCoef2_g, u_light_SHCoef0_b, u_light_SHCoef1_b, u_light_SHCoef2_b, u_lambertBRDF, u_transferFunc_ZHCoef, normal);\n\
  vec3 spec = specLighting(u_light_SHCoef0_r, u_light_SHCoef1_r, u_light_SHCoef2_r, u_light_SHCoef0_g, u_light_SHCoef1_g, u_light_SHCoef2_g, u_light_SHCoef0_b, u_light_SHCoef1_b, u_light_SHCoef2_b, u_lambertBRDF, view, normal, u_glossiness);\n\
  gl_FragColor = vec4(diffuse + spec, 1.0);\n\
}\n";




// Cube Map SH Coef
demoSHLightingExtractDLight.cubeMap_sh_coef= [9];
demoSHLightingExtractDLight.cubeMap_sh_coef[0]= Vector3(0.723579, 1.072841, 1.721793);
demoSHLightingExtractDLight.cubeMap_sh_coef[1]= Vector3(-0.060881, -0.067748, -0.055946);
demoSHLightingExtractDLight.cubeMap_sh_coef[2]= Vector3(0.035535, 0.034569, 0.141628);
demoSHLightingExtractDLight.cubeMap_sh_coef[3]= Vector3(-0.074809, -0.088456, -0.162581);
demoSHLightingExtractDLight.cubeMap_sh_coef[4]= Vector3(0.051700, 0.067442, 0.071479);
demoSHLightingExtractDLight.cubeMap_sh_coef[5]= Vector3(0.012779, 0.015840, 0.012821);
demoSHLightingExtractDLight.cubeMap_sh_coef[6]= Vector3(-0.257084, -0.338070, -0.475611);
demoSHLightingExtractDLight.cubeMap_sh_coef[7]= Vector3(-0.138954, -0.110993, -0.020484);
demoSHLightingExtractDLight.cubeMap_sh_coef[8]= Vector3(0.110703, 0.072421, -0.027866);


// max(NdotL, 0) ZH Coef
demoSHLightingExtractDLight.transferFunc_zh_coef= [3];
demoSHLightingExtractDLight.transferFunc_zh_coef[0]= (Math.sqrt(Math.PI)*0.5);
demoSHLightingExtractDLight.transferFunc_zh_coef[1]= Math.sqrt(Math.PI/3.0);
demoSHLightingExtractDLight.transferFunc_zh_coef[2]= Math.sqrt(5.0*Math.PI)/8.0;

demoSHLightingExtractDLight.shaderSkyBoxVS="attribute vec3  a_position;\n\
                                            attribute vec3  a_normal;\n\
                                            varying vec3    v_normal;\n\
                                            uniform mat4	u_viewProjMat;\n\
                                            void main() {\n\
                                                v_normal= a_normal;\n\
                                                gl_Position = u_viewProjMat * vec4(a_position.x, a_position.y, a_position.z, 1.0);\n\
                                            }\n";

demoSHLightingExtractDLight.shaderSkyBoxFS="precision highp float;\n\
                                            varying vec3    v_normal;\n\
                                            uniform samplerCube u_cubeMap;\n\
                                            void main() {\n\
                                                vec3 normal= normalize(v_normal);\n\
                                                gl_FragColor = textureCube(u_cubeMap, normal);\n\
                                            }\n";



// shader programs
demoSHLightingExtractDLight.shaderProgram_SHLighting= new Array(demoSHLightingExtractDLight.onOffEum.Num);
for(var i=0; i<demoSHLightingExtractDLight.onOffEum.Num; ++i){
    demoSHLightingExtractDLight.shaderProgram_SHLighting[i]= new Array(demoSHLightingExtractDLight.onOffEum.Num);
}
demoSHLightingExtractDLight.shaderProgram_skyBox= null;



// timer
demoSHLightingExtractDLight.timer= new Object();
demoSHLightingExtractDLight.timer.lastTime= (new Date()).getTime() * 0.001;
demoSHLightingExtractDLight.timer.elapsedTime = 0.0;
demoSHLightingExtractDLight.timer.totalTime = 0.0;
demoSHLightingExtractDLight.timer.update = function() {
    var now = (new Date()).getTime() * 0.001;
    this.elapsedTime = now - this.lastTime;
    this.lastTime = now;
    if (this.elapsedTime > 0.05)
        this.elapsedTime = 0.05; // clamp the time to avoid unstable simulation...
    this.totalTime += this.elapsedTime;
}

demoSHLightingExtractDLight.camPosOffset= Vector3(0.0, 0.0, 23.0);
demoSHLightingExtractDLight.camLookAt= Vector3(0.0, 0.0, 0.0);
demoSHLightingExtractDLight.camRotatePhi= 50.0;    // in degree
demoSHLightingExtractDLight.camRotateTheta= -5.0;   // in degree

var INF_VAL= 999999.9;

demoSHLightingExtractDLight.input= new Object();
demoSHLightingExtractDLight.input.isDragging= false;
demoSHLightingExtractDLight.input.isMoved= false;
demoSHLightingExtractDLight.input.isJustReleased= false;
demoSHLightingExtractDLight.input.mouseDownPos= new Object();
demoSHLightingExtractDLight.input.mouseDownPos.x= INF_VAL;
demoSHLightingExtractDLight.input.mouseDownPos.y= INF_VAL;

demoSHLightingExtractDLight.input.mouseCurrentPos= new Object();
demoSHLightingExtractDLight.input.mouseCurrentPos.x= INF_VAL;
demoSHLightingExtractDLight.input.mouseCurrentPos.y= INF_VAL;

demoSHLightingExtractDLight.input.mouseLastPos= new Object();
demoSHLightingExtractDLight.input.mouseLastPos.x= INF_VAL;
demoSHLightingExtractDLight.input.mouseLastPos.y= INF_VAL;

/**
 * The main entry point
 */
function demoSHLightingExtractDLight_main() {
	//
	demoSHLightingExtractDLight.canvas = document.getElementById("canvasSHLightingExtractDLight");
	demoSHLightingExtractDLight.gl = WebGLUtils.setupWebGL(demoSHLightingExtractDLight.canvas);
	//Couldn't setup GL
	if(!demoSHLightingExtractDLight.gl) {
		//alert("No WebGL!");
		return;
	}
    
    
	demoSHLightingExtractDLight.projectionMatrix= createPerspectiveProjectionMatrix(degreeToRadian(45.0), demoSHLightingExtractDLight.canvas.width/demoSHLightingExtractDLight.canvas.height, 0.1, 100.0);
    demoSHLightingExtractDLight.modelWorldMatrix= createRotationYMatrix(degreeToRadian(demoSHLightingExtractDLight.camRotatePhi));
    demoSHLightingExtractDLight.modelWorldMatrix[13]= -5.0;
    demoSHLightingExtractDLight.viewMatrix= createLookAtMatrix( 
        matrixMultiplyVector3(createRotationYMatrix(degreeToRadian(demoSHLightingExtractDLight.camRotatePhi)), matrixMultiplyVector3(createRotationXMatrix(degreeToRadian(demoSHLightingExtractDLight.camRotateTheta)), demoSHLightingExtractDLight.camPosOffset) )
        , demoSHLightingExtractDLight.camLookAt, Vector3(0.0, 1.0, 0.0) );
    
	//
	if(!demoSHLightingExtractDLight.init()) {
		alert("Could not init!");
		return;
	}

	//
	demoSHLightingExtractDLight.update();
}

function demoSHLightingExtractDLight_update() {
	demoSHLightingExtractDLight.update();
}

/**
 * Init our shaders, buffers and any additional setup
 */
demoSHLightingExtractDLight.init= function() {
	//
	if(!this.initShaders()) {
		alert("Could not init shaders!");
		return false;
	}

	//
	if(!this.initBuffers()) {
		alert("Could not init buffers!");
		return false;
	}

	//
	if(!this.initTextures()) {
		alert("Could not init textures!");
		return false;
	}

	//
	this.gl.clearColor(0.4, 0.4, 0.4, 1.0);
	this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
	this.gl.clearDepth(1.0);
    this.gl.enable(this.gl.DEPTH_TEST);
    this.gl.disable(this.gl.CULL_FACE);


	this.canvas.onmousedown= function(ev){
		demoSHLightingExtractDLight.input.mouseDownPos.x= ev.clientX;
		demoSHLightingExtractDLight.input.mouseDownPos.y= ev.clientY;
		demoSHLightingExtractDLight.canvas.onmousemove(ev);
		demoSHLightingExtractDLight.input.isDragging= true;
	}
	
	this.canvas.onmouseup= function(ev){
		demoSHLightingExtractDLight.input.mouseDownPos.x= INF_VAL;
		demoSHLightingExtractDLight.input.mouseDownPos.y= INF_VAL;
		demoSHLightingExtractDLight.canvas.onmousemove(ev);
		demoSHLightingExtractDLight.input.isDragging= false;
		demoSHLightingExtractDLight.input.isJustReleased= true;
	}
	
	this.canvas.onmousemove= function(ev){
		demoSHLightingExtractDLight.input.mouseLastPos.x= demoSHLightingExtractDLight.input.mouseLastPos.x== INF_VAL ? ev.clientX : demoSHLightingExtractDLight.input.mouseCurrentPos.x;
		demoSHLightingExtractDLight.input.mouseLastPos.y= demoSHLightingExtractDLight.input.mouseLastPos.y== INF_VAL ? ev.clientY : demoSHLightingExtractDLight.input.mouseCurrentPos.y;
		demoSHLightingExtractDLight.input.mouseCurrentPos.x= ev.clientX;
		demoSHLightingExtractDLight.input.mouseCurrentPos.y= ev.clientY;
		demoSHLightingExtractDLight.input.isMoved= true;
	}

	this.canvas.onmouseout= function(ev){
		if (demoSHLightingExtractDLight.input.isDragging){
			demoSHLightingExtractDLight.canvas.onmouseup(ev);
			demoSHLightingExtractDLight.input.isMoved= false;
		}
	}

	return true;
}

/**
 * Init our shaders, load them, create the program and attach them
 */
demoSHLightingExtractDLight.initShaders= function() {
    
    // SH lighting shader program
    {
        for (var i=0; i< demoSHLightingExtractDLight.onOffEum.Num; ++i)
        {
            for (var j=0; j< demoSHLightingExtractDLight.onOffEum.Num; ++j)
            {
                var createdShaderProgram= this.gl.createProgram();
                if(createdShaderProgram == null) {
                    alert("Cannot create Shader Program!");
                    return;
                }
                this.shaderProgram_SHLighting[i][j]= createdShaderProgram;
                
                var vertexShader = this.createShader(this.gl.VERTEX_SHADER, this.shaderSHLightingVS);
                var fsString= this.shaderSHLightingFS_def + this.shaderSHLightingFS_diffuse[i] + this.shaderSHLightingFS_spec[j] + this.shaderSHLightingFS_main;
                var fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fsString);
                
                this.gl.attachShader(createdShaderProgram, vertexShader);
                this.gl.attachShader(createdShaderProgram, fragmentShader);
                this.gl.linkProgram(createdShaderProgram);	
                
                if(!this.gl.getProgramParameter(createdShaderProgram, this.gl.LINK_STATUS)) {
                    alert("Could not link shader!");
                    this.gl.deleteProgram(createdShaderProgram);
                    return false;
                }
                
                this.gl.useProgram(this.shaderProgram);
                
                // get constant location
                createdShaderProgram.a_position = this.gl.getAttribLocation(createdShaderProgram, "a_position");
                createdShaderProgram.a_normal = this.gl.getAttribLocation(createdShaderProgram, "a_normal");
                this.gl.enableVertexAttribArray(createdShaderProgram.a_position);
                this.gl.enableVertexAttribArray(createdShaderProgram.a_normal);
                
                
                createdShaderProgram.u_glossiness           = this.gl.getUniformLocation(createdShaderProgram, "u_glossiness");
                createdShaderProgram.u_cameraPosition       = this.gl.getUniformLocation(createdShaderProgram, "u_cameraPosition");
                createdShaderProgram.u_worldMat             = this.gl.getUniformLocation(createdShaderProgram, "u_worldMat");
                createdShaderProgram.u_worldViewProjMat     = this.gl.getUniformLocation(createdShaderProgram, "u_worldViewProjMat");
                
                createdShaderProgram.u_light_SHCoef0_r      = this.gl.getUniformLocation(createdShaderProgram, "u_light_SHCoef0_r");
                createdShaderProgram.u_light_SHCoef1_r      = this.gl.getUniformLocation(createdShaderProgram, "u_light_SHCoef1_r");
                createdShaderProgram.u_light_SHCoef2_r      = this.gl.getUniformLocation(createdShaderProgram, "u_light_SHCoef2_r");
                
                createdShaderProgram.u_light_SHCoef0_g      = this.gl.getUniformLocation(createdShaderProgram, "u_light_SHCoef0_g");
                createdShaderProgram.u_light_SHCoef1_g      = this.gl.getUniformLocation(createdShaderProgram, "u_light_SHCoef1_g");
                createdShaderProgram.u_light_SHCoef2_g      = this.gl.getUniformLocation(createdShaderProgram, "u_light_SHCoef2_g");
                
                createdShaderProgram.u_light_SHCoef0_b      = this.gl.getUniformLocation(createdShaderProgram, "u_light_SHCoef0_b");
                createdShaderProgram.u_light_SHCoef1_b      = this.gl.getUniformLocation(createdShaderProgram, "u_light_SHCoef1_b");
                createdShaderProgram.u_light_SHCoef2_b      = this.gl.getUniformLocation(createdShaderProgram, "u_light_SHCoef2_b");
                
                createdShaderProgram.u_transferFunc_ZHCoef  = this.gl.getUniformLocation(createdShaderProgram, "u_transferFunc_ZHCoef");
                createdShaderProgram.u_lambertBRDF          = this.gl.getUniformLocation(createdShaderProgram, "u_lambertBRDF");
            }
        }
    }
    
    // Skybox shader program
    {
        var createdShaderProgram= this.gl.createProgram();
        if(createdShaderProgram == null) {
            alert("Cannot create Shader Program!");
            return;
        }
        this.shaderProgram_skyBox= createdShaderProgram;
        
        var vertexShader = this.createShader(this.gl.VERTEX_SHADER, this.shaderSkyBoxVS);
        var fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, this.shaderSkyBoxFS);
        
        this.gl.attachShader(createdShaderProgram, vertexShader);
        this.gl.attachShader(createdShaderProgram, fragmentShader);
        this.gl.linkProgram(createdShaderProgram);	
        
        if(!this.gl.getProgramParameter(createdShaderProgram, this.gl.LINK_STATUS)) {
            alert("Could not link shader!");
            this.gl.deleteProgram(createdShaderProgram);
            return false;
        }
        
        this.gl.useProgram(this.shaderProgram);
        
        // get constant location
        createdShaderProgram.a_position = this.gl.getAttribLocation(createdShaderProgram, "a_position");
        createdShaderProgram.a_normal = this.gl.getAttribLocation(createdShaderProgram, "a_normal");
        this.gl.enableVertexAttribArray(createdShaderProgram.a_position);
        this.gl.enableVertexAttribArray(createdShaderProgram.a_normal);
        
        createdShaderProgram.u_cubeMap= this.gl.getUniformLocation(createdShaderProgram, "u_cubeMap");
        createdShaderProgram.u_viewProjMat= this.gl.getUniformLocation(createdShaderProgram, "u_viewProjMat");
    }
    
	return true;
}

/**
 *
 */
demoSHLightingExtractDLight.createShader = function(shaderType, shaderSource) {
	//Create a shader
	var shader = this.gl.createShader(shaderType);
	//
	if(shader == null) {
		alert("Could not create shader!");
		return null;
	}

	//
	this.gl.shaderSource(shader, shaderSource);
	this.gl.compileShader(shader);

	//
	if(!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
		alert("Could not compile shader!" + this.gl.getShaderInfoLog(shader) );
		this.gl.deleteShader(shader);
		return null;
	}

	//
	return shader;
}

/**
 * Init our required buffers (in our case for our triangle)
 */
demoSHLightingExtractDLight.initBuffers= function() {
	// model
    {
        this.modelVertexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.modelVertexBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.modelVertex, this.gl.STATIC_DRAW);
    }
    
    // sky box
    {
        this.skyboxVertexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.skyboxVertexBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, this.skyboxVertex, this.gl.STATIC_DRAW);
    }
    
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, null);
    
	return true;
}

/**
 * Init textures for cube maps
 */
demoSHLightingExtractDLight.initTextures = function() {
	var faces= [6];
	faces[0]= this.gl.TEXTURE_CUBE_MAP_POSITIVE_X;
	faces[1]= this.gl.TEXTURE_CUBE_MAP_POSITIVE_Y;
	faces[2]= this.gl.TEXTURE_CUBE_MAP_POSITIVE_Z;
	faces[3]= this.gl.TEXTURE_CUBE_MAP_NEGATIVE_X;
	faces[4]= this.gl.TEXTURE_CUBE_MAP_NEGATIVE_Y;
	faces[5]= this.gl.TEXTURE_CUBE_MAP_NEGATIVE_Z;
    
	this.skyBoxCubeMap.handle= this.gl.createTexture();
	this.skyBoxCubeMap.images= [6];
	for (i=0; i<6; ++i){
		this.skyBoxCubeMap.images[i]= new Image();
		this.skyBoxCubeMap.images[i].handle= this.skyBoxCubeMap.handle;
		this.skyBoxCubeMap.images[i].face= faces[i];
		this.skyBoxCubeMap.images[i].onload= function(){
			demoSHLightingExtractDLight.textureImageLoaded(this, this.handle, this.face);
		}
		this.skyBoxCubeMap.images[i].src= this.skyBoxImageData[i];
	}
	return true;
}

demoSHLightingExtractDLight.textureImageLoaded= function(image, handle, face){
	this.gl.bindTexture(this.gl.TEXTURE_CUBE_MAP, handle);
	this.gl.texImage2D(face, 0, this.gl.RGB, this.gl.RGB, this.gl.UNSIGNED_BYTE, image);
	this.gl.texParameteri(this.gl.TEXTURE_CUBE_MAP, this.gl.TEXTURE_MAG_FILTER, this.gl.NEAREST);
	this.gl.texParameteri(this.gl.TEXTURE_CUBE_MAP, this.gl.TEXTURE_MIN_FILTER, this.gl.NEAREST);
	this.gl.bindTexture(this.gl.TEXTURE_CUBE_MAP, null);
}

/**
 * Update loop
 */
demoSHLightingExtractDLight.update= function(){

	this.timer.update();

    if ( document.getElementById('shLightingExtractDLight_isModelRotated').checked )
        this.modelWorldMatrix= matrixMultiplyMatrix(this.modelWorldMatrix, createRotationYMatrix( degreeToRadian(-15.0) * this.timer.elapsedTime) ) ;
    
	if (this.input.isDragging){
		// dragging
		var mouseDeltaX= this.input.mouseCurrentPos.x - this.input.mouseLastPos.x;
		var mouseDeltaY= this.input.mouseCurrentPos.y - this.input.mouseLastPos.y;
        
        this.camRotatePhi+= -mouseDeltaX;
        this.camRotateTheta+= -mouseDeltaY;
        if (this.camRotateTheta >= 89.0)
            this.camRotateTheta = 89.0;
        else if (this.camRotateTheta <= -89.0)
            this.camRotateTheta = -89.0;
	}
    
    this.camPos= matrixMultiplyVector3(createRotationYMatrix(degreeToRadian(this.camRotatePhi)), matrixMultiplyVector3(createRotationXMatrix(degreeToRadian(this.camRotateTheta)), this.camPosOffset) );
    this.viewMatrix= createLookAtMatrix( this.camPos, this.camLookAt, Vector3(0.0, 1.0, 0.0) );
    
	this.draw();

	// update input
	if (this.input.isMoved){
		this.input.mouseLastPos.x= this.input.mouseCurrentPos.x;
		this.input.mouseLastPos.y= this.input.mouseCurrentPos.y;
		this.input.isMoved= false;
	}
	if (this.input.isJustReleased)
		this.input.isJustReleased= false;
	window.requestAnimFrame(demoSHLightingExtractDLight_update, this.canvas);
}

/**
 * Our draw/render method
 */
demoSHLightingExtractDLight.draw= function() {	
	//
	this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
    
    var viewProj= matrixMultiplyMatrix(this.projectionMatrix, this.viewMatrix);
    var worldViewProj= matrixMultiplyMatrix(viewProj, this.modelWorldMatrix);

	// draw model
    {   
        var diffuseIdx= 0;
        if ( document.getElementById('shLightingExtractDLight_isRenderDiffuse').checked )
            diffuseIdx= this.onOffEum.on;
        else
            diffuseIdx= this.onOffEum.off;
        var specIdx= 0;
        if ( document.getElementById('shLightingExtractDLight_isRenderSpec').checked )
            specIdx= this.onOffEum.on;
        else
            specIdx= this.onOffEum.off;
        
        var shaderProgram= this.shaderProgram_SHLighting[diffuseIdx][specIdx];
        
        this.gl.useProgram(shaderProgram);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.modelVertexBuffer);
        this.gl.enableVertexAttribArray(shaderProgram.a_position);
        this.gl.enableVertexAttribArray(shaderProgram.a_normal);
        this.gl.vertexAttribPointer(shaderProgram.a_position, 3, this.gl.FLOAT, false, 4*6, 0);
        this.gl.vertexAttribPointer(shaderProgram.a_normal, 3, this.gl.FLOAT, false, 4*6, 4*3);
        
        var glossiness= parseFloat(document.getElementById("shLightingExtractDLight_glossiness").value);
        this.gl.uniform1f(shaderProgram.u_glossiness, glossiness);
        this.gl.uniform3f(shaderProgram.u_cameraPosition, this.camPos.x, this.camPos.y, this.camPos.z);
        
        this.gl.uniform3f(shaderProgram.u_light_SHCoef0_r, this.cubeMap_sh_coef[0].x, this.cubeMap_sh_coef[1].x, this.cubeMap_sh_coef[2].x);
        this.gl.uniform3f(shaderProgram.u_light_SHCoef1_r, this.cubeMap_sh_coef[3].x, this.cubeMap_sh_coef[4].x, this.cubeMap_sh_coef[5].x);
        this.gl.uniform3f(shaderProgram.u_light_SHCoef2_r, this.cubeMap_sh_coef[6].x, this.cubeMap_sh_coef[7].x, this.cubeMap_sh_coef[8].x);
        
        this.gl.uniform3f(shaderProgram.u_light_SHCoef0_g, this.cubeMap_sh_coef[0].y, this.cubeMap_sh_coef[1].y, this.cubeMap_sh_coef[2].y);
        this.gl.uniform3f(shaderProgram.u_light_SHCoef1_g, this.cubeMap_sh_coef[3].y, this.cubeMap_sh_coef[4].y, this.cubeMap_sh_coef[5].y);
        this.gl.uniform3f(shaderProgram.u_light_SHCoef2_g, this.cubeMap_sh_coef[6].y, this.cubeMap_sh_coef[7].y, this.cubeMap_sh_coef[8].y);
        
        this.gl.uniform3f(shaderProgram.u_light_SHCoef0_b, this.cubeMap_sh_coef[0].z, this.cubeMap_sh_coef[1].z, this.cubeMap_sh_coef[2].z);
        this.gl.uniform3f(shaderProgram.u_light_SHCoef1_b, this.cubeMap_sh_coef[3].z, this.cubeMap_sh_coef[4].z, this.cubeMap_sh_coef[5].z);
        this.gl.uniform3f(shaderProgram.u_light_SHCoef2_b, this.cubeMap_sh_coef[6].z, this.cubeMap_sh_coef[7].z, this.cubeMap_sh_coef[8].z);
        
        this.gl.uniform3f(shaderProgram.u_transferFunc_ZHCoef, this.transferFunc_zh_coef[0], this.transferFunc_zh_coef[1], this.transferFunc_zh_coef[2]);
        this.gl.uniform3f(shaderProgram.u_lambertBRDF, 1.0/(Math.PI), 1.0/(Math.PI), 1.0/(Math.PI));
        
        this.gl.uniformMatrix4fv(shaderProgram.u_worldMat, false, this.modelWorldMatrix);
        this.gl.uniformMatrix4fv(shaderProgram.u_worldViewProjMat, false, worldViewProj);
        this.gl.drawArrays(this.gl.TRIANGLES, 0, this.modelVertex.length/(3+3) );
    }

    // draw skybox
    {
        this.gl.enable(this.gl.TEXTURE_CUBE_MAP);
        this.gl.useProgram(this.shaderProgram_skyBox);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.skyboxVertexBuffer);
        this.gl.enableVertexAttribArray(this.shaderProgram_skyBox.a_position);
        this.gl.enableVertexAttribArray(this.shaderProgram_skyBox.a_normal);
        this.gl.vertexAttribPointer(this.shaderProgram_skyBox.a_position, 3, this.gl.FLOAT, false, 4*6, 0);
        this.gl.vertexAttribPointer(this.shaderProgram_skyBox.a_normal, 3, this.gl.FLOAT, false, 4*6, 4*3);
        
        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_CUBE_MAP, this.skyBoxCubeMap.handle );
        this.gl.uniform1i(this.shaderProgram_skyBox.u_cubeMap, 0);
        
        viewProj= matrixMultiplyMatrix(viewProj, createScaleMatrix(this.skyboxSize, this.skyboxSize, this.skyboxSize));
        this.gl.uniformMatrix4fv(this.shaderProgram_skyBox.u_viewProjMat, false, viewProj);
        this.gl.drawArrays(this.gl.TRIANGLES, 0, 240);
        this.gl.disable(this.gl.TEXTURE_CUBE_MAP);
    }

}

}
/*
     FILE ARCHIVED ON 20:22:48 Nov 17, 2019 AND RETRIEVED FROM THE
     INTERNET ARCHIVE ON 21:07:35 Aug 19, 2025.
     JAVASCRIPT APPENDED BY WAYBACK MACHINE, COPYRIGHT INTERNET ARCHIVE.

     ALL OTHER CONTENT MAY ALSO BE PROTECTED BY COPYRIGHT (17 U.S.C.
     SECTION 108(a)(3)).
*/
/*
playback timings (ms):
  captures_list: 0.502
  exclusion.robots: 0.015
  exclusion.robots.policy: 0.007
  esindex: 0.009
  cdx.remote: 67.249
  LoadShardBlock: 385.541 (3)
  PetaboxLoader3.datanode: 159.523 (4)
  PetaboxLoader3.resolve: 270.593 (2)
  load_resource: 116.556
*/