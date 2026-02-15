
const _isString = (arg) => typeof arg === 'string';

const _isArray = (arg) => Array.isArray(arg) || (ArrayBuffer.isView(arg) && !(arg instanceof DataView));

const _align16b = (n) => (n + 15) & 0xfffffff0;

/** class for ComputeShader Kernel*/
export class CSKernel {
    /** Create a CSKernel 
     *
     * @param {WebCS} webCS - instance of WebCS
     * @param {GPUShaderModule} prog - instance of GPUShaderModule  
     * @param {{}} settings - settings
     * @example
     * let settings = {
     *   "local_size":[8,8,1],
     *   "groups":[8,8,1],
     *   "params":{
     *       "all":["A","B","C"],
     *       "A":{"index":0,"type":{"type":"f32","dim":1}},
     *       "B":{"index":1,"type":{"type":"f32","dim":1}},
     *       "C":{"index":2,"type":{"type":"f32","dim":1}}},
     *   "uniform":{"MNK":{"type":"vec4<u32>","fields":{},"x":1,"y":1,"z":1,"index":0}}
     * };
     * let kernel = new CSKernel(webCS, prog, settings);
     */
    constructor(webCS, prog, settings = {}) {
        this.kernel = prog;
        this.local_size = settings.local_size || [32, 1, 1];
        this.groups = settings.groups;
        this.webCS = webCS;
        this.vids = null;
        this.computePipeline = null;
        this.settings = settings;
    }

    /**
     * Dispatch Compute Kernel
     *
     * @param {} arg -   a list of shader arguments, such as run(arg0, arg1, ..., argn)
     * @param {unfold_of_vec3} groups_xyz - optional, the size of work group, such as run(arg0, arg1, ..., argn, groups_x, groups_y, groups_z)
     * @param {{}} uniform    - optional, uniform , such as run(arg0, arg1, ..., argn, groups_x, groups_y, groups_z, {uniform_name:uniform[0,1,2,3]})
     * @example
     * run(arg0, arg1, ..., argn)
     * run(arg0, arg1, ..., argn, groups_x, groups_y, groups_z)
     * run(arg0, arg1, ..., argn, groups_x, groups_y, groups_z, {uniform_name:uniform[0,1,2,3]})
     * run(arg0, arg1, ..., argn, {uniform_name:uniform[0,1,2,3]})
     */
    async run(...args) {
        this.commandEncoder = this.webCS.gpuDevice.createCommandEncoder();
        this.__createPipeline();
        await this.__updateArgments(args);
        if (this.groups == null) {
            this.groups = [
                Math.floor(this.webCS.canvas.width / this.local_size[0]),
                Math.floor(this.webCS.canvas.height / this.local_size[1]), 1
            ];
        }
        const passEncoder = this.commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.computePipeline);
        if (this.bindGroup) {
            passEncoder.setBindGroup(0, this.bindGroup);
        }
        if (this.__getNumberOfUniform() > 0) {
            passEncoder.setBindGroup(1, this.uniformBindGroup);
        }
        passEncoder.dispatchWorkgroups(this.groups[0], this.groups[1], this.groups[2]);
        passEncoder.end();
        const gpuCommands = this.commandEncoder.finish();
        this.webCS.gpuDevice.queue.submit([gpuCommands]);
        await this.webCS.gpuDevice.queue.onSubmittedWorkDone();
    }

    setUniform(name, ...values) {
        const device = this.webCS.gpuDevice;
        const mytype = this.settings.uniform[name].type;
        const slot = this.settings.uniform[name].index;

        let uniformValue = null;
        const dataType = this.__sfmt2datatype(mytype);

        if (dataType === 'u32') {
            uniformValue = new Uint32Array(values);
        } else if (dataType === 'i32') {
            uniformValue = new Int32Array(values);
        } else if (dataType === 'f32') {
            uniformValue = new Float32Array(values);
        } else if (dataType === 'f16') {
            uniformValue = new Uint16Array(values);
        } else {
            uniformValue = new Uint32Array(values);
        }

        // TODO: get the correct length
        const bufferSizeInBytes = _align16b(4 * values.length);
        if (this.uniformVids[slot] == null) {
            this.uniformVids[slot] = device.createBuffer({
                mappedAtCreation: true,
                size: bufferSizeInBytes,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            const hostAccessBuffer = this.uniformVids[slot].getMappedRange();
            new uniformValue.constructor(hostAccessBuffer).set(uniformValue);
            this.uniformVids[slot].unmap();
        } else {
            device.queue.writeBuffer(
                this.uniformVids[slot], 0, uniformValue.buffer, uniformValue.byteOffset, uniformValue.byteLength);
        }
        return this;
    }

    getTexture(name) {
        return this.getBuffer(name);
    }

    getBuffer(name) {
        if (typeof name === 'string') {
            const index = this.settings.params.all.indexOf(name);
            return this.vids[index];
        } else if (typeof name === 'number') {
            return this.vids[name];
        }
    }

    async getData(name, dstarray) {
        const vid = this.getBuffer(name);
        return await this.webCS.getData(vid, dstarray);
    }

    setGroups(x, y = 1, z = 1) {
        this.groups = [x, y, z];
        return this;
    }

    __getNumberOfUniform() {
        return this.settings.uniform ? Object.keys(this.settings.uniform).length : 0;
    }

    __createPipeline() {
        if (this.computePipeline != null) return;
        this.__createLayout();
        const device = this.webCS.gpuDevice;
        const layouts = [];
        const { bindGroupLayout, uniformBindGroupLayout } = this;
        
        if (bindGroupLayout) {
            layouts.push(bindGroupLayout);
        }
        if (uniformBindGroupLayout) {
            layouts.push(uniformBindGroupLayout);
        }

        this.computePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
            compute: { module: this.kernel, entryPoint: 'main' }
        });
    }

    __createLayout() {
        const device = this.webCS.gpuDevice;
        const entries = [];
        
        for (let i = 0; i < this.settings.params.all.length; i++) {
            const argName = this.settings.params.all[i];
            const param = this.settings.params[argName];
            const argType = param.type;
            const argIndex = param.index;

            if (argType.dim === 1) {
                entries.push({ binding: argIndex, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
            } else if (argType.dim === 2) {
                if (param.rwmode === 'w') {
                    const sfmt = this.__str2sfmt(argType.type);
                    entries.push({
                        binding: argIndex,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: {
                            viewDimension: '2d',
                            access: 'write-only',
                            format: sfmt,
                        }
                    });
                } else {
                    entries.push({
                        binding: argIndex,
                        visibility: GPUShaderStage.COMPUTE,
                        texture: {
                            viewDimension: '2d',
                            sampleType: 'unfilterable-float',
                        }
                    });
                }
            } else {
                entries.push({ binding: argIndex, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
            }
        }
        
        if (entries.length > 0) {
            this.bindGroupLayout = device.createBindGroupLayout({ entries });
        }

        const uniform_entries = [];
        for (const uniform of Object.values(this.settings.uniform)) {
            uniform_entries.push(
                { binding: uniform.index, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
        }
        if (uniform_entries.length > 0) {
            this.uniformBindGroupLayout = device.createBindGroupLayout({ entries: uniform_entries });
        }
    }

    async __updateArg(i, arg) {
        const isBuffer = (argType) => argType.dim === 1;
        const isTexture = (argType) => argType.dim === 2;
        
        const device = this.webCS.gpuDevice;
        const argName = this.settings.params.all[i];
        const argType = this.settings.params[argName].type;

        if (isBuffer(argType)) {
            const w = this.groups[0] * this.settings.local_size[0];
            const h = this.groups[1] * this.settings.local_size[1];
            
            const createBuffer = (bytes) => {
                return device.createBuffer({
                    mappedAtCreation: true,
                    size: bytes,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
                });
            };

            if (arg == null) {
                if (this.vids[i] == null) {
                    const size = w * h * ((argType.type === 'double' ? 8 : 4));
                    const gpuBuffer = createBuffer(size);
                    gpuBuffer.unmap();
                    this.vids[i] = gpuBuffer;
                }
            } else if (arg instanceof GPUBuffer) {
                this.vids[i] = arg;
                // arg.unmap(); // Typically shouldn't unmap if passed in, unless we know it's mapped
            } else if (_isArray(arg)) {
                if (this.vids[i] != null) {
                    const vid_size = this.vids[i].size;
                    if (vid_size < arg.byteLength) {
                        this.vids[i] = createBuffer(arg.byteLength);
                        const hostAccessBuffer = this.vids[i].getMappedRange();
                        new arg.constructor(hostAccessBuffer).set(arg);
                        this.vids[i].unmap();
                    } else {
                        device.queue.writeBuffer(this.vids[i], 0, arg, 0);
                    }
                } else {
                    this.vids[i] = createBuffer(arg.byteLength);
                    const hostAccessBuffer = this.vids[i].getMappedRange();
                    new arg.constructor(hostAccessBuffer).set(arg);
                    this.vids[i].unmap();
                }
            }
        } else if (isTexture(argType)) {
            const sfmt = this.__str2sfmt(argType.type);
            
            const createTexture = (w, h, sfmt) => {
                w = w || this.webCS.canvas.width;
                h = h || this.webCS.canvas.height;
                sfmt = sfmt || 'rgba8unorm';
                const tex = device.createTexture({
                    size: { width: w, height: h },
                    format: sfmt,
                    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING |
                        GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
                });
                tex.size = [w, h, 4];
                return tex;
            };

            if (arg == null) {
                if (this.vids[i] == null) {
                    this.vids[i] = createTexture.call(this);
                }
            } else if (arg instanceof GPUTexture) {
                this.vids[i] = arg;
            } else if ((arg instanceof HTMLCanvasElement) || (arg instanceof HTMLImageElement)) {
                if (this.vids[i] == null) {
                    this.vids[i] = createTexture.call(this);
                }
                if (this.vids[i].width != null && this.vids[i].height != null &&
                    (this.vids[i].width < arg.width || this.vids[i].height < arg.height)) {
                    this.vids[i] = createTexture.call(this, arg.width, arg.height, sfmt);
                }
                const tex = this.vids[i];
                const imageBitmap = await createImageBitmap(arg);
                device.queue.copyExternalImageToTexture(
                    { source: imageBitmap }, { texture: tex }, [imageBitmap.width, imageBitmap.height]);
            }
        }
    }

    __sfmt2datatype(fmt) {
        return this.webCS.SFmt2DataType[fmt] || 'f32';
    }
    __sfmt2fmt(fmt) {
        return this.webCS.SFmt2Fmt[fmt] || 'rgba8unorm';
    }
    __str2sfmt(str) {
        return this.webCS.Str2SFmt[str] || 'rgba8unorm';
    }

    async __updateArgments(args) {
        const nargs = this.settings.params.all.length;
        this.vids = this.vids || Array.from({ length: nargs }, () => null);
        
        // Error checking for args length could be improved here
        
        for (let i = 0; i < this.vids.length; i++) {
            await this.__updateArg(i, args[i]);
        }

        if ((args.length === nargs + 3) || (args.length === nargs + 4)) {
            this.groups[0] = args[nargs];
            this.groups[1] = args[nargs + 1];
            this.groups[2] = args[nargs + 2];
        }

        const nUniform = Object.keys(this.settings.uniform).length;
        if (nUniform > 0) {
            this.uniformVids = this.uniformVids || Array.from({ length: nUniform }, () => null);
            if ((args.length === nargs + 1) || (args.length === nargs + 4)) {
                // last param is {'uniform_var':[]}
                const uniforms = args[args.length - 1];
                for (const uniform_key in uniforms) {
                    const uniform_args = [uniform_key].concat(uniforms[uniform_key]);
                    this.setUniform.apply(this, uniform_args);
                }
            }
        }
        this.__createBindGroup();
        this.__createUniformBindGroup();
    }

    __createBindGroup() {
        const device = this.webCS.gpuDevice;
        const bindGroupLayout = this.bindGroupLayout;
        const entries = [];
        
        for (let i = 0; i < this.settings.params.all.length; i++) {
            const buffer = this.vids[i];
            if (buffer instanceof GPUTexture) {
                entries.push({ binding: i, resource: buffer.createView() });
            } else {
                entries.push({ binding: i, resource: { buffer: buffer } });
            }
        }
        
        if (entries.length > 0) {
            this.bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries });
        }
    }

    __createUniformBindGroup() {
        const device = this.webCS.gpuDevice;
        const bindGroupLayout = this.uniformBindGroupLayout;
        const entries = [];
        const nUniform = Object.keys(this.settings.uniform).length;
        
        for (let i = 0; i < nUniform; i++) {
            const buffer = this.uniformVids[i];
            entries.push({ binding: i, resource: { buffer: buffer } });
        }
        
        if (entries.length > 0) {
            this.uniformBindGroup = device.createBindGroup({ layout: bindGroupLayout, entries });
        }
    }
}

/**
 *WebCS hosts the adapter, gpu device, canvas, and creates CSKernel
 * */
export class WebCS {
    /*private*/ constructor(adapter, device, settings = {}) {
        this.canvas = settings.canvas || document.createElement('canvas');
        if (settings.canvas == null) {
            const canvas = document.createElement('canvas');
            canvas.width = settings.width || 640;
            canvas.height = settings.height || 480;
            this.canvas = canvas;
        } else {
            this.canvas = settings.canvas;
        }
        this.adapter = adapter;
        this.gpuDevice = device;
        this.SFmt2DataType = {};
        this.SFmt2Fmt = {};
        this.Str2SFmt = {};
        this.presentSettings = { initialized: false };
        this.__setFmt();
    }

    /*
     * Create WebCs object from adapter and device
     */
    static async create(settings = {}) {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
            console.log('Failed to get GPU adapter.');
            return;
        }
        const device = await adapter.requestDevice();
        return new WebCS(adapter, device, settings);
    }

    createShaderFromString(source, settings = {}) {
        const shaderModule = this.gpuDevice.createShaderModule({ code: source });
        settings.commandEncoder = this.gpuDevice.createCommandEncoder();
        this.commandEncoder = settings.commandEncoder;
        return new CSKernel(this, shaderModule, settings);
    }

    createShaderFromFunction(func, settings = {}) {
        let csmain_str = func();
        let global_str = '';
        let global_func_str = (this.glsl_functions || '') + '\n' + (settings.glsl_functions || '');
        
        const isWhite = (ch) => ((ch === ' ') || (ch === '\t') || (ch === '\n'));
        const isSeperator = (ch) => ((ch === ' ') || (ch === '\t') || (ch === '\n') || (ch === '='));
        
        const comments = /(\/\/.*)|(\/\*[\s\S]*?\*\/)/g;
        let csmain_nocomments = csmain_str.replace(comments, '');

        // process shared
        const re_shared = /shared\s+[^;]+;/g;
        const matches = [...csmain_nocomments.matchAll(re_shared)];
        for (const match of matches) {
            global_str += match[0];
        }
        csmain_nocomments = csmain_nocomments.replace(re_shared, '');

        // process function
        let func_si = csmain_nocomments.indexOf('function');
        if (func_si > 0) {
            const indexOfendf = (str, si) => {
                let l = str.length;
                let ending = 0;
                for (let iii = si; iii < l; iii++) {
                    if (str[iii] === '{') ending++;
                    if (str[iii] === '}') {
                        ending--;
                        if (ending === 0) return iii + 1;
                    }
                }
                return null;
            };

            while (func_si > 0) {
                if (isWhite(csmain_nocomments[func_si + 8])) {
                    const funcEndI = indexOfendf(csmain_nocomments, func_si + 8);
                    if (funcEndI == null) {
                        // Error
                    }
                    global_func_str += '\nfn ' + csmain_nocomments.substring(func_si + 8, funcEndI);
                    csmain_nocomments = csmain_nocomments.substring(0, func_si) + csmain_nocomments.substring(funcEndI);
                    func_si = csmain_nocomments.indexOf('function');
                } else {
                    func_si = csmain_nocomments.indexOf('function', func_si + 8);
                }
            }
        }

        // process module-scope const
        func_si = csmain_nocomments.indexOf('const');
        if (func_si > 0) {
            while (func_si > 0) {
                if (isWhite(csmain_nocomments[func_si + 5])) {
                    const funcEndI = csmain_nocomments.indexOf(';', func_si + 5) + 1;
                    if (funcEndI == null) {
                        // Error
                    }
                    const myvar = 'const ' + csmain_nocomments.substring(func_si + 5, funcEndI);
                    global_func_str += '\n' + myvar;
                    csmain_nocomments = csmain_nocomments.substring(0, func_si) + csmain_nocomments.substring(funcEndI);
                    func_si = csmain_nocomments.indexOf('const');
                } else {
                    func_si = csmain_nocomments.indexOf('const', func_si + 5);
                }
            }
        }

        // process var<workgroup>
        func_si = csmain_nocomments.indexOf('var<workgroup>');
        if (func_si > 0) {
            while (func_si > 0) {
                const funcEndI = csmain_nocomments.indexOf(';', func_si + 14) + 1;
                if (funcEndI == null) {
                    // Error
                }
                const myvar = csmain_nocomments.substring(func_si, funcEndI);
                global_func_str += '\n' + myvar;
                csmain_nocomments = csmain_nocomments.substring(0, func_si) + csmain_nocomments.substring(funcEndI);
                func_si = csmain_nocomments.indexOf('var<workgroup>');
            }
        }

        // process parameters
        const func_str = func.toString();
        const startI = func_str.indexOf('(');
        const endI = func_str.indexOf(')');
        const param_str = func_str.substring(startI + 1, endI).replace(/\s/g, '');
        const params = param_str.split(',');
        
        if (settings.params == null) {
            settings.params = { all: params };
        } else {
            settings.params.all = params;
        }

        params.forEach((ele, idx) => {
            if (settings.params[ele] == null) {
                settings.params[ele] = { index: idx, type: this.__parsetype('buffer') };
            } else {
                settings.params[ele].index = idx;
            }
        });

        // parse declaration of param from wgsl
        params.forEach((ele) => {
            const myreg = new RegExp('var[\\s]+' + ele + '[\\s]*:[\\s]*([^;]+)[\\s]*;');
            const mymatch = csmain_nocomments.match(myreg);
            if (mymatch) {
                csmain_nocomments = csmain_nocomments.replace(myreg, '');
                settings.params[ele].type.final_type = mymatch[1];
            }
        });

        // process []
        const params_tex = Object.keys(settings.params)
            .filter(key => settings.params[key].type != null && settings.params[key].type.dim === 2);
            
        if (params_tex.length > 0) {
            for (let texname of params_tex) {
                const texreader2 = new RegExp(`(${texname})\\s*\\[([^\\[\\]]+)\\]\\s*\\[([^\\[\\]]+)\\]`, 'g');
                const texwriter2 = new RegExp(`(${texname})\\s*\\[([^\\[\\]]+)\\]\\s*\\[([^\\[\\]]+)\\]\\s=([^;]+);`, 'g');
                const texreader = new RegExp(`(${texname})\\s*\\[([^\\[\\]]+)\\]`, 'g');
                const texwriter = new RegExp(`(${texname})\\s*\\[([^\\[\\]]+)\\]\\s*=([^;]+);`, 'g');

                csmain_nocomments = csmain_nocomments.replace(texwriter2, 'textureStore($1,vec2<i32>(i32($3),i32($2)), $4);');
                csmain_nocomments = csmain_nocomments.replace(texwriter, 'textureStore($1,vec2<i32>($2), $3);');
                csmain_nocomments = csmain_nocomments.replace(texreader2, 'textureLoad($1,vec2<i32>(i32($3),i32($2)), 0);');
                csmain_nocomments = csmain_nocomments.replace(texreader, 'textureLoad($1,vec2<i32>($2), 0);');
            }

            const imgst_re = /textureStore\s*\(\s*([^,]+),/g;
            const matches = [...csmain_nocomments.matchAll(imgst_re)];
            for (const match of matches) {
                const vname = match[1].trim();
                if (settings.params[vname]) {
                    settings.params[vname].rwmode = 'w';
                }
            }
        }
        
        for (let paramname of settings.params.all) {
            const memaccessor = new RegExp(`[\\W](${paramname})\\s*\\[`, 'g');
            csmain_nocomments = csmain_nocomments.replace(memaccessor, ' $1.data[');
        }

        let layout_str = '';
        for (let pi = 0; pi < params.length; pi++) {
            const param_name = params[pi];
            const param_type = settings.params[param_name].type;
            if (param_type.dim === 1) { // buffer
                const num_type = param_type.type;
                const final_type = param_type.final_type || `array<${num_type}>`;
                layout_str += ` struct struct_${param_name}{ data: ${final_type}} ;\n@group(0) @binding(${pi}) var<storage, read_write> ${param_name} : struct_${param_name};\n`;
            } else if (param_type.dim === 2) { // texture
                const pix_type = this.__str2sfmt(param_type);
                const value_type = this.__sfmt2datatype(pix_type);
                const rwmode = settings.params[param_name].rwmode === 'w' ? 'writeonly' : 'readonly';
                let attr = settings.params[param_name].attr || "";
                
                if ((attr.indexOf('readonly') === -1) && (attr.indexOf('writeonly') === -1)) {
                    attr += " " + rwmode; 
                }
                
                if (attr.indexOf('readonly') > 0) {
                    const final_type = param_type.final_type || `texture_2d<${value_type}>`;
                    layout_str += `@group(0) @binding(${pi}) var ${param_name} : ${final_type};\n`;
                } else {
                    const final_type = param_type.final_type || `texture_storage_2d<${pix_type}, write>`;
                    layout_str += `@group(0) @binding(${pi}) var ${param_name} : ${final_type};\n`;
                }
            }
        }

        // process uniforms
        let unform_str = '';
        if (!settings.hasOwnProperty('uniform')) {
            settings.uniform = {};
        }

        const re = /this\.uniform\.([a-zA-Z0-9_-]{1,})\.([a-zA-Z0-9_-]{1,})/g;
        const re2 = /this\.uniform\.([a-zA-Z0-9_-]{1,})([^\.a-zA-Z0-9_-])+/g;
        const matches_uniform = [...csmain_nocomments.matchAll(re)];
        const matches2_uniform = [...csmain_nocomments.matchAll(re2)];
        
        const indexOfNonSpace = (s, startIndex) => {
            let si = startIndex;
            while (isWhite(s[si])) si++;
            return si;
        };
        const indexOfSeperator = (s, startIndex) => {
            let si = startIndex;
            while (!isSeperator(s[si])) si++;
            return si;
        };

        const types = {
            'u32': 'vec4<u32>',
            'f32': 'vec4<f32>',
            'i32': 'vec4<i32>',
            'f64': 'vec4<f64>',
            'vec4<i32>': 'vec4<i32>',
            'vec4<u32>': 'vec4<u32>',
            'vec4<f32>': 'vec4<f32>',
            'vec4<f64>': 'vec4<f64>'
        };

        const updateOneMatch = (match) => {
            let lineStartI = 0;
            const vname = match[1];
            lineStartI = Math.max(lineStartI, csmain_nocomments.lastIndexOf(';', match.index));
            lineStartI = Math.max(lineStartI, csmain_nocomments.lastIndexOf('}', match.index));
            lineStartI = Math.max(lineStartI, csmain_nocomments.lastIndexOf('{', match.index));
            lineStartI = lineStartI + 1;
            
            const colonIndex = csmain_nocomments.lastIndexOf(':', match.index);
            if (colonIndex > lineStartI) {
                const type_si = indexOfNonSpace(csmain_nocomments, colonIndex + 1);
                const type_ei = indexOfSeperator(csmain_nocomments, type_si);
                const type_str = csmain_nocomments.substring(type_si, type_ei);
                const mytype = types[type_str] || type_str;
                settings.uniform[vname]['type'] = mytype;
            }
        };

        for (const match of matches_uniform) {
            const vname = match[1];
            if (settings.uniform[vname] == null) {
                settings.uniform[vname] = { type: null, fields: {} };
            }
            settings.uniform[vname][match[2]] = 1;
            updateOneMatch(match);
        }

        for (const match of matches2_uniform) {
            const vname = match[1];
            if (settings.uniform[vname] == null) {
                settings.uniform[vname] = { type: null, fields: {} };
            }
            updateOneMatch(match);
        }

        let pi = 0;
        for (const uniform in settings.uniform) {
            const mytype = settings.uniform[uniform].type || 'vec4<u32>';
            settings.uniform[uniform].index = pi;
            const my_uniform_str = `
            struct struct_${uniform} {data:${mytype}};
            @group(1) @binding(${pi}) var<uniform> ${uniform} : struct_${uniform};`;
            unform_str += my_uniform_str;
            pi++; // Increment pi for binding
        }

        for (const uniform in settings.uniform) {
            const memaccessor = new RegExp(`this.uniform.(${uniform})`, 'g');
            csmain_nocomments = csmain_nocomments.replace(memaccessor, '$1.data');
        }

        const local_size = settings.local_size || [8, 8, 1];
        
        const source = `
        ${layout_str} 
        ${unform_str}
        ${global_str}
        const LOCAL_SIZE_X:u32 = ${local_size[0]}u;
        const LOCAL_SIZE_Y:u32 = ${local_size[1]}u;
        const LOCAL_SIZE_Z:u32 = ${local_size[2]}u;
        var<private> g_num_workgroups:vec3<u32>;
        var<private> g_workgroup_id:vec3<u32>;
        ${global_func_str}
        fn csmain(thread: vec3<u32>, localthread:vec3<u32>, workgroup_id:vec3<u32>){
            ${csmain_nocomments}
        }
        @compute @workgroup_size(${local_size[0]}, ${local_size[1]}, ${local_size[2]})

        fn main(@builtin(global_invocation_id) thread: vec3<u32>, @builtin(local_invocation_id) localthread: vec3<u32>, @builtin(workgroup_id) block: vec3<u32>, @builtin(num_workgroups) wgs:vec3<u32>) {
            g_num_workgroups = wgs;
            g_workgroup_id = block;
            csmain(thread, localthread, block);
        }
        `;
        
        return this.createShaderFromString(source, settings);
    }

    __parsetype(type) {
        type = type.replace(/\s*/g, '');
        let bi = -1;
        let t = 'f32';
        let dim = 1;
        
        if (type === 'buffer') {
            t = 'f32';
            dim = 1;
        } else if (type === 'texture') {
            t = 'rgba8unorm';
            dim = 2;
        } else if ((bi = type.indexOf('[][]')) >= 0) {
            t = (bi === 0) ? 'rgba8unorm' : type.substring(0, bi).toLowerCase();
            dim = 2;
        } else if ((bi = type.indexOf('[]')) >= 0) {
            t = (bi === 0) ? 'f32' : type.substring(0, bi).toLowerCase();
            if (['f32', 'u32', 'i32'].includes(t)) {
                // ok
            } else if (t === 'float') {
                t = 'f32';
            } else {
                throw new Error('type is not supported : ' + t);
            }
            dim = 1;
        }
        return { 'type': t, 'dim': dim };
    }

    createShader(source, settings = {}) {
        if (settings.params != null) {
            for (const key in settings.params) {
                const v = settings.params[key];
                if (_isString(v)) {
                    settings.params[key] = { type: this.__parsetype(v) };
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

    async present(tex) {
        const { canvas, gpuDevice: device } = this;
        const context = canvas.getContext('webgpu');
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat(); 

        if (this.presentSettings.initialized === false) {
            const presentationSize = [canvas.width, canvas.height];
            context.configure({
                device,
                format: presentationFormat,
                size: presentationSize,
            });
            const fullscreenTexturedQuadWGSL = `
        @group(0) @binding(0) var mSampler : sampler;
        @group(0) @binding(1) var mTexture : texture_2d<f32>;

        struct VertexOutput {
            @builtin(position) Position : vec4<f32>,
            @location(0) fragUV : vec2<f32>
        };

        @vertex
        fn vert_main(@builtin(vertex_index) VertexIndex : u32) -> VertexOutput {
            var pos = array<vec2<f32>, 4>(
                vec2<f32>( 1.0,  1.0),
                vec2<f32>( 1.0, -1.0),
                vec2<f32>(-1.0,  1.0),
                vec2<f32>(-1.0,  -1.0));

            var uv = array<vec2<f32>, 4>(
                vec2<f32>(1.0, 0.0),
                vec2<f32>(1.0, 1.0),
                vec2<f32>(0.0, 0.0),
                vec2<f32>(0.0, 1.0));

            var output : VertexOutput;
            output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
            output.fragUV = uv[VertexIndex];
            return output;
        }

        @fragment
        fn frag_main(@location(0) fragUV : vec2<f32>) -> @location(0) vec4<f32> {
            var color:vec4<f32> = textureSample(mTexture, mSampler, fragUV);
            return color;
        }
        `;
            const bindGroupLayout = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                    { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} }
                ]
            });
            const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

            const fullscreenQuadPipeline = device.createRenderPipeline({
                layout: pipelineLayout, 
                vertex: {
                    module: device.createShaderModule({ code: fullscreenTexturedQuadWGSL }),
                    entryPoint: 'vert_main',
                },
                fragment: {
                    module: device.createShaderModule({ code: fullscreenTexturedQuadWGSL }),
                    entryPoint: 'frag_main',
                    targets: [{ format: presentationFormat }],
                },
                primitive: { topology: 'triangle-strip' },
            });
            const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

            this.presentSettings.sampler = sampler;
            this.presentSettings.fullscreenQuadPipeline = fullscreenQuadPipeline;
            this.presentSettings.initialized = true;
        }

        const commandEncoder = device.createCommandEncoder();
        const { sampler, fullscreenQuadPipeline } = this.presentSettings;
        
        const renderBindGroup = device.createBindGroup({
            layout: fullscreenQuadPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: sampler },
                { binding: 1, resource: tex.createView() },
            ],
        });
        
        const passEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 1.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        
        passEncoder.setPipeline(fullscreenQuadPipeline);
        passEncoder.setBindGroup(0, renderBindGroup);
        passEncoder.draw(4, 1, 0, 0);
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);
    }

    createBuffer(size) {
        const device = this.gpuDevice;
        const gpuBuffer = device.createBuffer({
            mappedAtCreation: true,
            size: size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        return gpuBuffer;
    }

    createTexture(fmt = 'rgba8unorm', w, h) {
        w = w || this.canvas.width;
        h = h || this.canvas.height;

        const device = this.gpuDevice;
        const tex = device.createTexture({
            size: { width: w, height: h },
            format: fmt,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });
        tex.size = [w, h, 4];
        return tex;
    }

    async getData(vid, dstarray) {
        const isTypedArray = (arr) => ArrayBuffer.isView(arr) && !(arr instanceof DataView);
        
        const getHostAccessArrary = (dstarray, arrayBuffer) => {
            if (dstarray == undefined) {
                return new Uint8Array(arrayBuffer);
            } else if (typeof dstarray === 'string') {
                const typemap = new Map([
                    ['int8', Int8Array], ['uint8', Uint8Array], ['uint8clamped', Uint8ClampedArray],
                    ['int16', Int16Array], ['uint16', Uint16Array], ['int32', Int32Array], ['uint32', Uint32Array],
                    ['float32', Float32Array], ['float', Float32Array], ['float64', Float64Array],
                    ['bigint', BigInt64Array], ['bigint', BigUint64Array]
                ]);
                const TypedArray = typemap.get(dstarray) || Uint8Array;
                return new TypedArray(arrayBuffer);
            } else if (isTypedArray(dstarray)) {
                return new dstarray.constructor(arrayBuffer);
            } else {
                return new Uint8Array(arrayBuffer);
            }
        };

        const resultBufferSizeInBytes = vid.size || 64 * 64 * 4;

        const gpuReadBuffer = this.gpuDevice.createBuffer(
            { size: resultBufferSizeInBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        const commandEncoder = this.gpuDevice.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            vid, 0, gpuReadBuffer, 0, resultBufferSizeInBytes
        );

        const gpuCommands = commandEncoder.finish();
        this.gpuDevice.queue.submit([gpuCommands]);

        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = gpuReadBuffer.getMappedRange();
        const hostAccessArrary = getHostAccessArrary(dstarray, arrayBuffer);

        if (isTypedArray(dstarray) && (dstarray.length === hostAccessArrary.length)) {
            dstarray.set(hostAccessArrary);
        } else {
            dstarray = hostAccessArrary.slice();
        }
        gpuReadBuffer.unmap();
        return dstarray;
    }

    __setFmt() {
        const fmts = [
            ['rgba8unorm' , 'rgba8unorm' , 'f32', 'rgba8'],
            ['rgba8unorm' , 'rgba8unorm' , 'f32', 'rgba'],
            ['rgba8unorm' , 'rgba8unorm' , 'f32', 'rgba8unorm'],
            ['rgba8snorm' , 'rgba8snorm' , 'f32', 'rgba8snorm'],
            ['rgba8uint'  , 'rgba8uint'  , 'u32', 'rgba8uint'],
            ['rgba8sint'  , 'rgba8sint'  , 'i32', 'rgba8sint'],
            ['rgba16uint' , 'rgba16uint' , 'u32', 'rgba16uint'],
            ['rgba16sint' , 'rgba16sint' , 'i32', 'rgba16sint'],
            ['rgba16float', 'rgba16float', 'f32', 'rgba16float'],
            ['r32uint'    , 'r32uint'    , 'u32', 'r32uint'],
            ['r32sint'    , 'r32sint'    , 'i32', 'r32sint'],
            ['r32float'   , 'r32float'   , 'f32', 'r32float'],
            ['rg32uint'   , 'rg32uint'   , 'u32', 'rg32uint'],
            ['rg32sint'   , 'rg32sint'   , 'i32', 'rg32sint'],
            ['rg32float'  , 'rg32float'  , 'f32', 'rg32float'],
            ['rgba32uint' , 'rgba32uint' , 'u32', 'rgba32uint'],
            ['rgba32sint' , 'rgba32sint' , 'i32', 'rgba32sint'],
            ['rgba32float', 'rgba32float', 'f32', 'rgba32float'],
            ['u32'        , 'u32'        , 'u32', 'u32'],
            ['i32'        , 'i32'        , 'i32', 'i32'],
            ['f32'        , 'f32'        , 'f32', 'f32'],
            ['f16'        , 'f16'        , 'f16', 'f16'],
            ['vec2i'      , 'vec2i'      , 'i32', 'vec2<i32>'],
            ['vec2<i32>'  , 'vec2i'      , 'i32', 'vec2<i32>'],
            ['vec2u'      , 'vec2u'      , 'u32', 'vec2<u32>'],
            ['vec2<u32>'  , 'vec2u'      , 'u32', 'vec2<u32>'],
            ['vec2f'      , 'vec2f'      , 'f32', 'vec2<f32>'],
            ['vec2<f32>'  , 'vec2f'      , 'f32', 'vec2<f32>'],
            ['vec2h'      , 'vec2h'      , 'f16', 'vec2<f16>'],
            ['vec2<f16>'  , 'vec2h'      , 'f16', 'vec2<f16>'],
            ['vec3i'      , 'vec3i'      , 'i32', 'vec3<i32>'],
            ['vec3<i32>'  , 'vec3i'      , 'i32', 'vec3<i32>'],
            ['vec3u'      , 'vec3u'      , 'u32', 'vec3<u32>'],
            ['vec3<u32>'  , 'vec3u'      , 'u32', 'vec3<u32>'],
            ['vec3f'      , 'vec3f'      , 'f32', 'vec3<f32>'],
            ['vec3<f32>'  , 'vec3f'      , 'f32', 'vec3<f32>'],
            ['vec3h'      , 'vec3h'      , 'f16', 'vec3<f16>'],
            ['vec3<f16>'  , 'vec3h'      , 'f16', 'vec3<f16>'],
            ['vec4i'      , 'vec4i'      , 'i32', 'vec4<i32>'],
            ['vec4<i32>'  , 'vec4i'      , 'i32', 'vec4<i32>'],
            ['vec4u'      , 'vec4u'      , 'u32', 'vec4<u32>'],
            ['vec4<u32>'  , 'vec4u'      , 'u32', 'vec4<u32>'],
            ['vec4f'      , 'vec4f'      , 'f32', 'vec4<f32>'],
            ['vec4<f32>'  , 'vec4f'      , 'f32', 'vec4<f32>'],
            ['vec4h'      , 'vec4h'      , 'f16', 'vec4<f16>'],
            ['vec4<f16>'  , 'vec4h'      , 'f16', 'vec4<f16>']
        ];

        for (const [sfmt, fmt, dt, str] of fmts) {
            this.SFmt2Fmt[sfmt] = fmt;
            this.SFmt2DataType[sfmt] = dt;
            this.Str2SFmt[str] = sfmt;
        }
    }
}
