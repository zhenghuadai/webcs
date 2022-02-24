(function(win) {
function _isString(arg) {
    return typeof arg === 'string';
}
function _isArray(arg) {
    return Array.isArray(arg) ||
        (ArrayBuffer.isView(arg) && !(arg instanceof DataView));
}
class CSKernel {
    constructor(prog, settings = {}) {
        this.kernel = prog;
        this.local_size = settings.local_size || [32, 1, 1];
        this.groups = settings.groups;
        this.webCS = settings.webCS || new WebCS();
        this.vids = null;
        this.settings = settings;
    }
    run() {
        var gl = this.webCS.gl;
        gl.useProgram(this.kernel);
        this.updateArgments(arguments);
        if (this.groups == null) {
            this.groups = [
                Math.floor(this.webCS.canvas.width / this.local_size[0]),
                Math.floor(this.webCS.canvas.height / this.local_size[1]), 1
            ];
        }
        gl.dispatchCompute(this.groups[0], this.groups[1], this.groups[2]);
        gl.memoryBarrier(gl.SHADER_STORAGE_BARRIER_BIT)
        gl.memoryBarrier(gl.SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }
    __updateArg(i, arg) {
        let isBuffer = function(argType) {
            return argType.dim == 1;
        };
        let isTexture = function(argType) {
            return argType.dim == 2;
        };
        var gl = this.webCS.gl;
        let argName = this.settings.params.all[i];
        let argType = this.settings.params[argName].type;
        if (isBuffer(argType)) {
            let w = this.settings.groups[0] * this.settings.local_size[0];
            let h = this.settings.groups[1] * this.settings.local_size[1];
            if (arg == null) {
                if (this.vids[i] == null) {
                    let size = w * h * ((argType.type == 'double' ? 8 : 4));
                    this.vids[i] = gl.createBuffer();
                    gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, this.vids[i]);
                    gl.bufferData(
                        gl.SHADER_STORAGE_BUFFER, size, gl.STATIC_DRAW);
                    gl.bindBufferBase(
                        gl.SHADER_STORAGE_BUFFER, i, this.vids[i]);
                }
            } else if (arg instanceof WebGLBuffer) {
                gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, i, arg);
                this.vids[i] = arg;
            } else if (_isArray(arg)) {
                if (this.vids[i] != null) {
                    gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, this.vids[i]);
                    let vid_size = gl.getBufferParameter(
                        gl.SHADER_STORAGE_BUFFER, gl.BUFFER_SIZE);
                    if (vid_size < arg.byteLength) {
                        // gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, null);
                        gl.deleteBuffer(this.vids[i]);
                        this.vids[i] = gl.createBuffer();
                    }
                } else {
                    this.vids[i] = gl.createBuffer();
                }
                gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, this.vids[i]);
                gl.bufferData(gl.SHADER_STORAGE_BUFFER, arg, gl.STATIC_DRAW);
                gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, i, this.vids[i]);
            }
        } else if (isTexture(argType)) {
            let sfmt = this.__str2sfmt(argType.type);
            let fmt = this.__sfmt2fmt(sfmt);
            let dataType = this.__fmt2datatype(sfmt);
            function createTexture(w, h) {
                w = w || this.webCS.canvas.width;
                h = h || this.webCS.canvas.height;
                let tex = gl.createTexture();
                gl.bindTexture(gl.TEXTURE_2D, tex);
                gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
                tex.width = w;
                tex.height = h;
                return tex;
            };
            if (arg == null) {
                if (this.vids[i] == null) {
                    this.vids[i] = createTexture.apply(this);
                }
            } else if (arg instanceof WebGLTexture) {
                this.vids[i] = arg;
            } else if (
                (arg instanceof HTMLCanvasElement) ||
                (arg instanceof HTMLImageElement)) {
                if (this.vids[i] == null) {
                    this.vids[i] = createTexture.apply(this);
                }
                if (this.vids[i].width != null && this.vids[i].height != null &&
                    (this.vids[i].width < arg.width ||
                     this.vids[i].height < arg.height)) {
                    this.vids[i] =
                        createTexture.apply(this, [arg.width, arg.height]);
                }
                let w = arg.width;
                let h = arg.height;
                gl.bindTexture(gl.TEXTURE_2D, this.vids[i]);
                gl.texSubImage2D(
                    gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, dataType, arg);
            }
            gl.bindImageTexture(
                i, this.vids[i], 0, false, 0, gl.READ_WRITE, gl.RGBA8);
        } else {
            // error
        }
    }

    __fmt2datatype(fmt) {
        return this.webCS.Fmt2DataType[fmt] || gl.UNSIGNED_BYTE;
    }
    __sfmt2fmt(fmt) {
        return this.webCS.SFmt2Fmt[fmt] || gl.RGBA;
    }
    __str2sfmt(str) {
        return this.webCS.Str2sFmt[str] || gl.RGBA8;
    }

    setUniform(name, v0, ...rest) {
        var gl = this.webCS.gl;
        gl.useProgram(this.kernel);
        let loc = gl.getUniformLocation(this.kernel, name);
        let values = [v0, 0, 0, 0];
        for (let i = 0; i < rest.length; i++) {
            values[i + 1] = rest[i];
        }
        let mytype = this.settings.uniform[name].type;
        if (mytype == 'uvec4') {
            gl.uniform4ui(loc, v0, values[1], values[2], values[3]);
        } else if (mytype == 'vec4') {
            gl.uniform4f(loc, v0, values[1], values[2], values[3]);
        }
        return this;
    }

    getTexture(name) {
        return this.getBuffer(name);
    }
    getBuffer(name) {
        if (typeof name === 'string') {
            let findIndex = function(o, value) {
                return o.indexOf(value);
            };
            let index = findIndex(this.settings.params.all, name);
            return this.vids[index];
        } else if (typeof name === 'number') {
            return this.vids[name];
        }
    }

    bindBuffer(name) {
        var gl = this.webCS.gl;
        let vid = this.getBuffer(name);
        gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, vid);
    }

    getData(name, dstarray) {
        var gl = this.webCS.gl;
        let vid = this.getBuffer(name);
        return this.webCS.getData(vid, dstarray);
    }

    setGroups(x, y = 1, z = 1) {
        this.groups = [x, y, z];
        return this;
    }

    // run(arg0, arg1, ..., argn)
    // run(arg0, arg1, ..., argn, groups_x, groups_y, groups_z)
    // run(arg0, arg1, ..., argn, groups_x, groups_y, groups_z, {uniform_name:uniform[0,1,2,3]})
    // run(arg0, arg1, ..., argn, {uniform_name:uniform[0,1,2,3]})
    updateArgments(args) {
        let nargs = this.settings.params.all.length;
        this.vids = this.vids || Array.from({length: nargs}, (v, i) => null);
        if ((args.lenght != nargs) && (args.length != nargs + 1) &&
            (args.length != nargs + 3) && (args.length != nargs + 4)) {
            // error
        }
        for (var i = 0; i < this.vids.length; i++) {
            this.__updateArg(i, args[i]);
        }
        if ((args.length == nargs + 3) || (args.length == nargs + 4)) {
            this.groups[0] = args[nargs];
            this.groups[1] = args[nargs + 1];
            this.groups[2] = args[nargs + 2];
        }
        if ((args.length == nargs + 1) || (args.length == nargs + 4)) {
            // last param is {'uniform_var':[]}
            let uniforms = args[args.length - 1];
            for (var uniform_key in uniforms) {
                let uniform_args = [uniform_key].concat(uniforms[uniform_key]);
                this.setUniform.apply(this, uniform_args);
            }
        }
    }
};
var csmain = `
uvec3 thread;
void main() {
    thread = gl_GlobalInvocationID;   
    csmain();
}
`
class WebCS {
    constructor(settings = {}) {
        this.canvas = settings.canvas || document.createElement('canvas');
        if (settings.canvas == null) {
            let canvas = document.createElement('canvas');
            canvas.width = settings.width || 640;
            canvas.height = settings.height || 480;
            this.canvas = canvas;
        } else {
            this.canvas = settings.canvas;
        }
        this.gl = this.canvas.getContext('webgl2-compute', {antialias: false});
        this.Fmt2DataType = {};
        this.SFmt2Fmt = {};
        this.Str2sFmt = {};
        this.__setFmt();
    }
    __setFmt() {
        let gl = this.gl;
        let fmts = [
            [gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, 'rgba8'],
            [gl.RGBA32F, gl.RGBA, gl.FLOAT, 'rgba32f'],
            [gl.RGBA32I, gl.RGBA, gl.INT, 'rgba32i'],
            [gl.RGBA32UI, gl.RGBA, gl.UNSIGNED_INT, 'rgba32ui'],
        ];
        for (let fmt of fmts) {
            this.Fmt2DataType[fmt[0]] = fmt[2];
            this.SFmt2Fmt[fmt[0]] = fmt[1];
            this.Str2sFmt[fmt[3]] = fmt[0];
        }
    }
    createShaderFromString(source, settings = {}) {
        let local_size = settings.local_size || [8, 8, 1];
        // clang-format off
        let local_size_str = `#version 310 es
        #define LOCAL_SIZE_X ${local_size[0]}u
        #define LOCAL_SIZE_Y ${local_size[1]}u
        #define LOCAL_SIZE_Z ${local_size[2]}u
        #define thread gl_GlobalInvocationID
        #define localthread gl_LocalInvocationID
        layout (local_size_x = ${local_size[0]}, local_size_y = ${local_size[1]}, local_size_z = ${local_size[2]}) in;`
        // clang-format on 
        var gl = this.gl;
        var shader = gl.createShader(gl.COMPUTE_SHADER);
        gl.shaderSource(shader, local_size_str + source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            // error
            var message = gl.getShaderInfoLog(shader);
            console.log(message);
            settings.error = message;
        }

        var cProg = gl.createProgram()
        gl.attachShader(cProg, shader)
        gl.linkProgram(cProg)
        if (!gl.getProgramParameter(cProg, gl.LINK_STATUS)) {
            // error
            console.log(gl.getProgramInfoLog(cProg));
        }
        settings.webCS = this;
        return new CSKernel(cProg, settings);
    }
    createShaderFromFunction(func, settings = {}) {
        let csmain_str = func();
        var gl = this.gl;
        let layout_str = '';
        let comments = /(\/\/.*)|(\/\*[\s\S]*?\*\/)/g;
        let csmain_nocomments = csmain_str.replace(comments, '');
        let global_str = "";
        let global_func_str = ( this.glsl_functions || "" ) + "\n"
        + (settings.glsl_functions || "");
        var isWhite = function(ch) {
            return ((ch == ' ') || (ch == '\t') || (ch == '\n'));
        };
        // process the shared 
        if(true)
        {
            let re_shared = /shared\s+[^;]+;/g;
            let matches = [...csmain_nocomments.matchAll(re_shared)];
            for (let match of matches) {
                global_str = global_str + match[0];
            }
            csmain_nocomments = csmain_nocomments.replace(re_shared, "");
        }

        // porcess the function
        if(true){
            let func_si = csmain_nocomments.indexOf('function');
            if(func_si > 0){
                function indexOfendf(str, si){
                    let l = str.length;
                    let ending = 0;
                    for(let iii = si; iii < l; iii++){
                        if(str[iii] == '{') ending = ending +1;
                        if(str[iii] == '}'){
                            ending = ending - 1;
                            if(ending == 0){
                                return iii+1;
                            }
                        }
                    }
                    return null;
                }
                while(func_si > 0){
                    if(isWhite(csmain_nocomments[func_si+8])){
                        let funcEndI = indexOfendf(csmain_nocomments, func_si+8);
                        if(funcEndI == null){
                            // Error 
                        }
                        global_func_str = global_func_str + '\n' + csmain_nocomments.substring(func_si+8, funcEndI);
                        csmain_nocomments = csmain_nocomments.substring(0,func_si) + csmain_nocomments.substring(funcEndI);
                        func_si = csmain_nocomments.indexOf('function');
                    }else{
                        func_si = csmain_nocomments.indexOf('function', func_si+8);
                    }
                }
            }
        }

        // process the parameters
        if (true) {
            let func_str = func.toString();
            let startI = func_str.indexOf('(');
            let endI = func_str.indexOf(')');
            let param_str =
                func_str.substring(startI + 1, endI).replace(/\s/g, '');
            let params = param_str.split(',');
            if (settings.params == null) {
                settings.params = {all: params};
            } else {
                settings.params.all = params;
            }
            if (true) {
                let that = this;
                params.forEach(function(ele, idx) {
                    if (settings.params[ele] == null) {
                        settings.params[ele] = {index: idx, type: that.__parsetype('buffer')};
                    } else {
                        settings.params[ele].index = idx;
                    }
                });
            }

            // procecss [], detect readonly/writeonly
            if (true) {
                let params_tex = Object.keys(settings.params)
                        .filter(key => settings.params[key].type != null && settings.params[key].type.dim === 2);
                if (0 == params_tex.length ) {
                } else {
                    // process the texture[]
                    if(true){
                        for (let texname of  params_tex) {
                            let texreader2 = new RegExp(['(', texname , ')',  '\\s*\\[([^\\[\\]]+)\\]\s*\\[([^\\[\\]]+)\\]'].join(''), 'g');
                            let texwriter2 = new RegExp(['(', texname , ')',  '\\s*\\[([^\\[\\]]+)\\]\\s*\\[([^\\[\\]]+)\\]\\s=([^;]+);'].join(''), 'g');
                            let texreader = new RegExp(['(', texname , ')',  '\\s*\\[([^\\[\\]]+)\\]'].join(''), 'g');
                            let texwriter = new RegExp(['(', texname , ')',  '\\s*\\[([^\\[\\]]+)\\]\\s*=([^;]+);'].join(''), 'g');
                            //let texreaders = [...csmain_nocomments.matchAll(texreader)];
                            //let texwriters = [...csmain_nocomments.matchAll(texwriter)];
                            csmain_nocomments= csmain_nocomments.replace(texwriter2, 'imageStore($1,ivec2($3,$2), $4);');
                            csmain_nocomments= csmain_nocomments.replace(texwriter, 'imageStore($1,ivec2($2), $3);');
                            csmain_nocomments= csmain_nocomments.replace(texreader2, 'imageLoad($1,ivec2($3,$2));');
                            csmain_nocomments= csmain_nocomments.replace(texreader, 'imageLoad($1,ivec2($2));');
                        }
                    }
                    // let's find out the imageStore
                    if(true){
                        let imgst_re = /imageStore\s*\(\s*([^,]+),/g;
                        let matches = [...csmain_nocomments.matchAll(imgst_re)];
                        for (let match of matches) {
                            var vname = match[1].trim();
                            if (settings.params[vname] == null) {
                                // error
                            } else {
                                settings.params[vname].rwmode = 'w';
                            }
                        }
                    }
                }
            }

            // Declare the params in GLSL
            for (var pi = 0; pi < params.length; pi++) {
                let param_name = params[pi];
                let param_type = settings.params[param_name].type;
                if (param_type.dim == 1) { // buffer
                    let num_type = param_type.type;
                    // clang-format off
                     layout_str = layout_str +
                         `layout (std430, binding = ${pi}) buffer ssb${param_name} {  ${num_type} ${param_name}[]; };`;
                    // clang-format on
                } else if (param_type.dim == 2) {  // texture
                    // clang-format off
                    let pix_type = param_type.type;
                    let rwmode = settings.params[param_name].rwmode == 'w' ?  'writeonly' : 'readonly';
                    let attr = settings.params[param_name].attr || "";
                    if((attr.indexOf('readonly') == -1 ) && (attr.indexOf('writeonly') == -1)){
                        attr = attr + " " + rwmode; 
                    }
                    layout_str = layout_str +
                        `layout (${pix_type}, binding = ${pi}) ${attr} uniform highp image2D ${param_name};`
                    // clang-format on
                } else {
                    // error
                }
            }
        }
        // process the uniforms
        let unform_str = '';
        if (true) {
            settings.uniform = {};
            let re = /this\.uniform\.([a-zA-Z0-9_-]{1,})\.([a-zA-Z0-9_-]{1,})/g;
            let re2 = /this\.uniform\.([a-zA-Z0-9_-]{1,})([^\.a-zA-Z0-9_-])+/g;
            let matches = [...csmain_nocomments.matchAll(re)];
            let matches2 = [...csmain_nocomments.matchAll(re2)];
            let revar = /[a-zA-Z0-9_-]{1,}/
            var indexOfSpace = function(s, startIndex) {
                let si = startIndex;
                while (!isWhite(s[si])) si++;
                return si;
            };
            var indexOfNonSpace = function(s, startIndex) {
                let si = startIndex;
                while (isWhite(s[si])) si++;
                return si;
            };
            var types = {
                'uint': 'uvec4',
                'float': 'vec4',
                'int': 'ivec4',
                'double': 'dvec4',
                'ivec4': 'ivec4',
                'uvec4': 'uvec4',
                'vec4': 'vec4',
                'dvec4': 'dvec4'
            };
            for (let match of matches) {
                var vname = match[1];
                if (settings.uniform[vname] == null) {
                    settings.uniform[vname] = { type: null, fields: {} }
                }
                settings.uniform[vname][match[2]] = 1;
                if (true) {
                    let lineStartI = 0;
                    lineStartI = Math.max(
                        lineStartI,
                        csmain_nocomments.lastIndexOf(';', match.index));
                    lineStartI = Math.max(
                        lineStartI,
                        csmain_nocomments.lastIndexOf('}', match.index));
                    lineStartI = Math.max(
                        lineStartI,
                        csmain_nocomments.lastIndexOf('{', match.index));
                    lineStartI = lineStartI + 1;
                    let type_si =
                        indexOfNonSpace(csmain_nocomments, lineStartI);
                    let type_ei = indexOfSpace(csmain_nocomments, type_si);
                    let type_str =
                        csmain_nocomments.substring(type_si, type_ei);
                    let mytype = types[type_str] || 'vec4';
                    settings.uniform[vname]['type'] = mytype;
                }
            }
            for (let match of matches2) {
                var vname = match[1];
                if (settings.uniform[vname] == null) {
                    settings.uniform[vname] = { type: null, fields: {} }
                }
            }
            for (let uniform in settings.uniform) {
                let mytype = settings.uniform[uniform].type;
                let my_uniform_str = ` uniform ${mytype} ${uniform};
                        `;
                unform_str = unform_str + my_uniform_str;
            }
            csmain_nocomments =
                csmain_nocomments.replace(/this\.uniform\./g, '');
        }

        // clang-format off
        let source = `
        ${layout_str} 
        ${unform_str}
        ${global_str}
        ${global_func_str}
        void csmain(){
            ${csmain_nocomments}
        }
        void main() {
            csmain();
        }
        `
        // clang-format on
        return this.createShaderFromString(source, settings);
    }

    __parsetype(type) {
        type = type.replace(/\s*/g, '');
        let bi = -1;
        let t = 'float';
        let dim = 1;
        if (type == 'buffer') {
            t = 'float';
            dim = 1;
        } else if (type == 'texture') {
            t = 'rgba8';
            dim = 2;
        } else if ((bi = type.indexOf('[][]')) >= 0) {
            t = (bi == 0) ? 'rgba8' : type.substring(0, bi).toLowerCase();
            dim = 2;
        } else if ((bi = type.indexOf('[]')) >= 0) {
            t = (bi == 0) ? 'float' : type.substring(0, bi).toLowerCase();
            dim = 1;
        } else {
            // error
        }
        return {'type': t, 'dim': dim};
    }
    createShader(source, settings = {}) {
        if (true) {
            // convert settings.params
            if (settings.params != null) {
                for (let key in settings.params) {
                    let v = settings.params[key];
                    if (_isString(v)) {
                        settings.params[key] = {type: this.__parsetype(v)};
                    }
                }
            }
        }
        if (typeof source === 'string') {
            return this.createShaderFromString(source, settings);
        } else {
            return this.createShaderFromFunction(source, settings);
        }
    }

    addFunctions(func) {
        this.glsl_functions = this.glsl_functions || '';
        this.glsl_functions += func;
    }

    present(tex) {
        let gl = this.gl;
        const frameBuffer = gl.createFramebuffer();
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, frameBuffer);
        gl.framebufferTexture2D(
            gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        let w = this.canvas.width;
        let h = this.canvas.height;
        gl.blitFramebuffer(
            0, 0, w, h, 0, h, w, 0, gl.COLOR_BUFFER_BIT, gl.NEAREST);
    }
    createBuffer(size) {
        let gl = this.gl;
        let buffer = gl.createBuffer();
        gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, buffer);
        gl.bufferData(gl.SHADER_STORAGE_BUFFER, size, gl.STATIC_DRAW);
        return buffer;
    }
    createTexture(fmt, w, h) {
        let gl = this.gl;
        fmt = fmt || gl.RGBA8;
        w = w || this.canvas.width;
        h = h || this.canvas.height;
        let tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, w, h);
        tex.width = w;
        tex.height = h;
        return tex;
    }
    getData(vid, dstarray) {
        var gl = this.gl;
        gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, vid);
        let vid_size =
            gl.getBufferParameter(gl.SHADER_STORAGE_BUFFER, gl.BUFFER_SIZE);
        if (dstarray == undefined) {
            dstarray = new Uint8Array(vid_size);
        } else if (typeof dstarray === 'string') {
            if (dstarray === 'float') {
                dstarray = new Float32Array(vid_size / 4);
            } else if (dataarray === 'uint32') {
                dstarray = new Uint32Array(vid_size / 4);
            } else if (dataarray === 'uint8') {
                dstarray = new Uint8Array(vid_size / 4);
            } else {
                dstarray = new Uint8Array(vid_size / 4);
            }
        }
        gl.getBufferSubData(gl.SHADER_STORAGE_BUFFER, 0, dstarray)
        return dstarray;
    }
};
win.WebCS = WebCS;
})(window);
