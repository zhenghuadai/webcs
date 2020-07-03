(function(win) {
class CSKernel {
    constructor(prog, settings = {}) {
        this.kernel = prog;
        this.local_size = settings.local_size || [32, 1, 1];
        this.groups = settings.groups || [1, 1, 1];
        this.webCS = settings.webCS || new WebCS();
        this.vids = null;
        this.settings = settings;
    }
    run() {
        var gl = this.webCS.gl;
        gl.useProgram(this.kernel);
        this.updateArgments(arguments);
        gl.dispatchCompute(this.groups[0], this.groups[1], this.groups[2]);
        gl.memoryBarrier(gl.SHADER_STORAGE_BARRIER_BIT)
    }
    __updateArg(i, arg) {
        var gl = this.webCS.gl;
        if (arg instanceof WebGLBuffer) {
            gl.bindBufferBase(gl.SHADER_STORAGE_BUFFER, i, arg);
            this.vids[i] = arg;
        } else if (
            Array.isArray(arg) ||
            (ArrayBuffer.isView(arg) && !(arg instanceof DataView))) {
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
    }

    setUniform(name, v0, ...rest) {
        var gl = this.webCS.gl;
        gl.useProgram(this.kernel);
        let loc = gl.getUniformLocation(this.kernel, name);
        let values = [v0].concat(rest);
        let mytype = this.settings.uniform[name].type;
        if (mytype == 'uvec4') {
            gl.uniform4ui(loc, v0, values[1], values[2], values[3]);
        } else if (mytype == 'vec4') {
            gl.uniform4f(loc, v0, values[1], values[2], values[3]);
        }
        return this;
    }

    getBuffer(name) {
        if (typeof name === 'string') {
            let findIndex = function(o, value) {
                for (var key in o) {
                    if (o[key] == value) {
                        return key;
                    }
                }
                return null;
            };
            let index = findIndex(this.settings.params, name);
            return this.vids[index];
        } else if (typeof name === 'number') {
            return this.vids[name];
        }
    }

    bindBuffer(name){
        var gl = this.webCS.gl;
        let vid = this.getBuffer(name);
        gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, vid);
    }

    getData(name, dstarray) {
        var gl = this.webCS.gl;
        let vid = this.getBuffer(name);
        gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, vid);
        let vid_size =
            gl.getBufferParameter(gl.SHADER_STORAGE_BUFFER, gl.BUFFER_SIZE);
        if (dstarray == undefined) {
            dstarray = new Uint8Array(vid_size);
        } else if (typeof dstarray === 'string') {
            if (dstarray === 'float') {
                dstarray = new Float32Array(vid_size / 4);
            }
        }
        gl.getBufferSubData(gl.SHADER_STORAGE_BUFFER, 0, dstarray)
        return dstarray;
    }

    updateArgments(args) {
        this.vids =
            this.vids || Array.from({length: args.length}, (v, i) => null);
        if (args.lenght != this.vids.length) {
            // error
        }
        for (var i = 0; i < args.length; i++) {
            this.__updateArg(i, args[i]);
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
        this.gl = this.canvas.getContext('webgl2-compute')
    }
    createShaderFromString(source, settings = {}) {
        let local_size = settings.local_size;
        let local_size_str = `#version 310 es
        layout (local_size_x = ${local_size[0]}, local_size_y = ${
            local_size[1]}, local_size_z = ${local_size[2]}) in;`
        var gl = this.gl;
        var shader = gl.createShader(gl.COMPUTE_SHADER);
        gl.shaderSource(shader, local_size_str + source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            // error
        }

        var cProg = gl.createProgram()
        gl.attachShader(cProg, shader)
        gl.linkProgram(cProg)
        if (!gl.getProgramParameter(cProg, gl.LINK_STATUS)) {
            // error
        }
        settings.webCS = this;
        return new CSKernel(cProg, settings);
    }
    createShaderFromFunction(func, settings = {}) {
        let csmain_str = func();
        var gl = this.gl;
        let layout_str = '';
        // process the parameters
        if (true) {
            let func_str = func.toString();
            let startI = func_str.indexOf('(');
            let endI = func_str.indexOf(')');
            let param_str =
                func_str.substring(startI + 1, endI).replace(/\s/g, '');
            let params = param_str.split(',');
            settings.params = {};
            for (var pi = 0; pi < params.length; pi++) {
                layout_str = layout_str +
                    `layout (std430, binding = ${pi}) buffer ssb${
                                 params[pi]} {  float ${params[pi]}[]; };`;
                settings.params[pi] = [params[pi]];
            }
        }
        // process the uniforms
        let unform_str = '';
        if (true) {
            settings.uniform = {};
            let re = /this\.uniform\.([a-zA-Z0-9_-]{1,})\.([a-zA-Z0-9_-]{1,})/g;
            let re2 = /this\.uniform\.([a-zA-Z0-9_-]{1,})([^\.a-zA-Z0-9_-])+/g;
            let comments = /(\/\/.*)|(\/\*[\s\S]*?\*\/)/g;
            let csmain_nocomments = csmain_str.replace(comments, '');
            let matches = [...csmain_nocomments.matchAll(re)];
            let matches2 = [...csmain_nocomments.matchAll(re2)];
            let revar = /[a-zA-Z0-9_-]{1,}/
            var isWhite = function(ch) {
                return ((ch == ' ') || (ch == '\t') || (ch == '\n'));
            };
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
                let my_uniform_str = `uniform ${mytype} ${uniform};`;
                unform_str = unform_str + my_uniform_str;
            }
            csmain_str = csmain_str.replace(/this\.uniform\./g, '');
        }

        let source = `
        ${layout_str} 
        ${unform_str}
        uvec3 thread;
        void csmain(){
            ${csmain_str}
        }
        void main() {
            thread = gl_GlobalInvocationID;   
            csmain();
        }
        `
        return this.createShaderFromString(source, settings);
    }
    createShader(source, settings = {}) {
        if (typeof source === 'string') {
            return this.createShaderFromString(source, settings);
        } else {
            return this.createShaderFromFunction(source, settings);
        }
    }
};
win.WebCS = WebCS;
})(window);
