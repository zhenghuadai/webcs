(function(win) {
function _isString(arg)
{
    return typeof arg === 'string';
}
function _isArray(arg)
{
    return Array.isArray(arg) || (ArrayBuffer.isView(arg) && !(arg instanceof DataView));
}
/** class for ComputeShader Kernel*/
class CSKernel
{
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
    constructor(webCS, prog, settings = {})
    {
        this.kernel          = prog;
        this.local_size      = settings.local_size || [32, 1, 1];
        this.groups          = settings.groups;
        this.webCS           = webCS;
        this.vids            = null;
        this.computePipeline = null;
        this.settings        = settings;
    };

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
    async run()
    {
        this.commandEncoder = this.webCS.gpuDevice.createCommandEncoder();
        this.__createPipeline();
        await this.__updateArgments(arguments);
        if (this.groups == null)
        {
            this.groups = [
                Math.floor(this.webCS.canvas.width / this.local_size[0]),
                Math.floor(this.webCS.canvas.height / this.local_size[1]), 1
            ];
        }
        const passEncoder = this.commandEncoder.beginComputePass();
        passEncoder.setPipeline(this.computePipeline);
        if (this.bindGroup)
        {
            passEncoder.setBindGroup(0, this.bindGroup);
        }
        if (this.__getNumberOfUniform() > 0)
        {
            console.log(this.uniformBindGroup)
            passEncoder.setBindGroup(1, this.uniformBindGroup);
        }
        passEncoder.dispatchWorkgroups(this.groups[0], this.groups[1], this.groups[2]);
        passEncoder.end();
        const gpuCommands = this.commandEncoder.finish();
        this.webCS.gpuDevice.queue.submit([gpuCommands]);
        await this.webCS.gpuDevice.queue.onSubmittedWorkDone();
    };
    setUniform(name, ...rest)
    {
        let device = this.webCS.gpuDevice;
        let values = rest; // [v0, 0, 0, 0];
        //for (let i = 0; i < rest.length; i++)
        //{
        //    values[i + 1] = rest[i];
        //}
        let mytype = this.settings.uniform[name].type;
        const slot = this.settings.uniform[name].index;

        let uniformValue = null;
        if (mytype == 'vec4<u32>')
        {
            uniformValue = new Uint32Array(values);
        }
        else if (mytype == 'vec4<i32>')
        {
            uniformValue = new int32Array(values);
        }
        else if (mytype == 'vec4<f32>')
        {
            uniformValue = new Float32Array(values);
        }
        else
        {
            uniformValue = new Uint32Array(values);
        }

        let bufferSizeInBytes = 16;
        if (this.uniformVids[slot] == null)
        {
            this.uniformVids[slot] = device.createBuffer({
                mappedAtCreation: true,
                size: bufferSizeInBytes,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });
            const hostAccessBuffer = this.uniformVids[slot].getMappedRange();
            new uniformValue.constructor(hostAccessBuffer).set(uniformValue);
            this.uniformVids[slot].unmap();
        }
        else
        {
            device.queue.writeBuffer(
                this.uniformVids[slot], 0, uniformValue.buffer, uniformValue.byteOffset, uniformValue.byteLength);
        }
        return this;
    };

    getTexture(name)
    {
        return this.getBuffer(name);
    };
    getBuffer(name)
    {
        if (typeof name === 'string')
        {
            let findIndex = function(o, value) {
                return o.indexOf(value);
            };
            let index = findIndex(this.settings.params.all, name);
            return this.vids[index];
        }
        else if (typeof name === 'number')
        {
            return this.vids[name];
        }
    };

    async getData(name, dstarray)
    {
        let vid = this.getBuffer(name);
        return await this.webCS.getData(vid, dstarray);
    };

    setGroups(x, y = 1, z = 1)
    {
        this.groups = [x, y, z];
        return this;
    };

    __getNumberOfUniform()
    {
        return this.settings.uniform ? Object.keys(this.settings.uniform).length : 0;
    };

    __createPipeline()
    {
        if (this.computePipeline != null)
            return;
        this.__createLayout();
        let device                 = this.webCS.gpuDevice;
        let layouts                = [];
        let bindGroupLayout        = this.bindGroupLayout;
        let uniformBindGroupLayout = this.uniformBindGroupLayout;
        if (bindGroupLayout)
        {
            layouts.push(bindGroupLayout);
        }
        if (uniformBindGroupLayout)
        {
            layouts.push(uniformBindGroupLayout);
        }

        this.computePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: layouts }),
            compute: { module: this.kernel, entryPoint: 'main' }
        });
    };
    __createLayout()
    {
        let device  = this.webCS.gpuDevice;
        let entries = [];
        for (var i = 0; i < this.settings.params.all.length; i++)
        {
            let argName  = this.settings.params.all[i];
            let param    = this.settings.params[argName];
            let argType  = param.type;
            let argIndex = param.index;
            if (argType.dim == 1)
            {
                entries.push({ binding: argIndex, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
            }
            else if (argType.dim == 2)
            {
                if (param.rwmode == 'w')
                {
                    let sfmt = this.__str2sfmt(argType.type);
                    entries.push({
                        binding: argIndex,
                        visibility: GPUShaderStage.COMPUTE,
                        storageTexture: {
                            viewDimension: '2d',
                            access: 'write-only',
                            format: sfmt,
                        }
                    });
                }
                else
                {
                    entries.push({
                        binding: argIndex,
                        visibility: GPUShaderStage.COMPUTE,
                        texture: {
                            viewDimension: '2d',
                            sampleType: 'unfilterable-float',
                        }
                    });
                }
            }
            else
            {
                entries.push({ binding: argIndex, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } });
            }
        }
        if (entries.length > 0)
        {
            this.bindGroupLayout = device.createBindGroupLayout({ entries: entries });
        }

        let uniform_entries = [];
        for (const [key, uniform] of Object.entries(this.settings.uniform))
        {
            uniform_entries.push(
                { binding: uniform.index, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } });
        }
        if (uniform_entries.length > 0)
        {
            this.uniformBindGroupLayout = device.createBindGroupLayout({ entries: uniform_entries });
        }
    };
    async __updateArg(i, arg)
    {
        let isBuffer = function(argType) {
            return argType.dim == 1;
        };
        let isTexture = function(argType) {
            return argType.dim == 2;
        };
        let device  = this.webCS.gpuDevice;
        let argName = this.settings.params.all[i];
        let argType = this.settings.params[argName].type;
        if (isBuffer(argType))
        {
            let w = this.settings.groups[0] * this.settings.local_size[0];
            let h = this.settings.groups[1] * this.settings.local_size[1];
            function createBuffer(bytes)
            {
                let gpuBuffer  = device.createBuffer({
                    mappedAtCreation: true,
                    size: bytes,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
                });
                gpuBuffer.size = bytes;
                return gpuBuffer;
            }
            if (arg == null)
            {
                if (this.vids[i] == null)
                {
                    let size = w * h * ((argType.type == 'double' ? 8 : 4));

                    let gpuBuffer          = createBuffer(size);
                    const hostAccessBuffer = gpuBuffer.getMappedRange();
                    //new Float32Array(hostAccessBuffer).set(firstMatrix);
                    gpuBuffer.unmap();
                    this.vids[i] = gpuBuffer;
                }
            }
            else if (arg instanceof GPUBuffer)
            {
                this.vids[i] = arg;
                arg.unmap();
            }
            else if (_isArray(arg))
            {
                if (this.vids[i] != null)
                {
                    let vid_size = this.vids[i].size;
                    if (vid_size < arg.byteLength)
                    {
                        // gl.bindBuffer(gl.SHADER_STORAGE_BUFFER, null);
                        this.vids[i]           = createBuffer(arg.byteLength);
                        const hostAccessBuffer = this.vids[i].getMappedRange();
                        new Float32Array(hostAccessBuffer).set(arg);
                        this.vids[i].unmap();
                    }
                    else
                    {
                        //device.queue.writeBuffer(this.vids[i], 0, arg, 0, arg.byteLength);
                        device.queue.writeBuffer(this.vids[i], 0, arg, 0);
                    }
                }
                else
                {
                    this.vids[i]           = createBuffer(arg.byteLength);
                    const hostAccessBuffer = this.vids[i].getMappedRange();
                    new Float32Array(hostAccessBuffer).set(arg);
                    this.vids[i].unmap();
                }
            }
        }
        else if (isTexture(argType))
        {
            let sfmt     = this.__str2sfmt(argType.type);
            let fmt      = this.__sfmt2fmt(sfmt);
            let dataType = this.__sfmt2datatype(sfmt);
            function createTexture(w, h, sfmt)
            {
                w        = w || this.webCS.canvas.width;
                h        = h || this.webCS.canvas.height;
                sfmt     = sfmt || 'rgba8unorm';
                let tex  = device.createTexture({
                    size: {
                        width: w,
                        height: h,
                    },
                    format: sfmt,
                    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING |
                        GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
                });
                tex.size = [w, h, 4];
                return tex;
            };
            if (arg == null)
            {
                if (this.vids[i] == null)
                {
                    this.vids[i] = createTexture.apply(this);
                }
            }
            else if (arg instanceof GPUTexture)
            {
                this.vids[i] = arg;
            }
            else if ((arg instanceof HTMLCanvasElement) || (arg instanceof HTMLImageElement))
            {
                if (this.vids[i] == null)
                {
                    this.vids[i] = createTexture.apply(this);
                }
                if (this.vids[i].width != null && this.vids[i].height != null &&
                    (this.vids[i].width < arg.width || this.vids[i].height < arg.height))
                {
                    this.vids[i] = createTexture.apply(this, [arg.width, arg.height, sfmt]);
                }
                let tex         = this.vids[i];
                let w           = arg.width;
                let h           = arg.height;
                let imageBitmap = await createImageBitmap(arg);
                device.queue.copyExternalImageToTexture(
                    { source: imageBitmap }, { texture: tex }, [imageBitmap.width, imageBitmap.height]);
            }
        }
        else
        {
            // error
        }
    };

    __sfmt2datatype(fmt)
    {
        return this.webCS.SFmt2DataType[fmt] || 'f32';
    };
    __sfmt2fmt(fmt)
    {
        return this.webCS.SFmt2Fmt[fmt] || 'rgba8unorm';
    };
    __str2sfmt(str)
    {
        return this.webCS.Str2SFmt[str] || 'rgba8unorm';
    };

    // run(arg0, arg1, ..., argn)
    // run(arg0, arg1, ..., argn, groups_x, groups_y, groups_z)
    // run(arg0, arg1, ..., argn, groups_x, groups_y, groups_z, {uniform_name:uniform[0,1,2,3]})
    // run(arg0, arg1, ..., argn, {uniform_name:uniform[0,1,2,3]})
    async __updateArgments(args)
    {
        let nargs = this.settings.params.all.length;
        this.vids = this.vids || Array.from({ length: nargs }, (v, i) => null);
        if ((args.lenght != nargs) && (args.length != nargs + 1) && (args.length != nargs + 3) &&
            (args.length != nargs + 4))
        {
            // error
        }
        for (var i = 0; i < this.vids.length; i++)
        {
            await this.__updateArg(i, args[i]);
        }
        if ((args.length == nargs + 3) || (args.length == nargs + 4))
        {
            this.groups[0] = args[nargs];
            this.groups[1] = args[nargs + 1];
            this.groups[2] = args[nargs + 2];
        }
        let nUniform = Object.keys(this.settings.uniform).length;
        if (Object.keys(this.settings.uniform).length > 0)
        {
            this.uniformVids = this.uniformVids || Array.from({ length: nUniform }, (v, i) => null);
            if ((args.length == nargs + 1) || (args.length == nargs + 4))
            {
                // last param is {'uniform_var':[]}
                let uniforms = args[args.length - 1];
                for (var uniform_key in uniforms)
                {
                    let uniform_args = [uniform_key].concat(uniforms[uniform_key]);
                    this.setUniform.apply(this, uniform_args);
                }
            }
        }
        this.__createBindGroup();
        this.__createUniformBindGroup();
    };
    __createBindGroup()
    {
        let device          = this.webCS.gpuDevice;
        let bindGroupLayout = this.bindGroupLayout;
        let entries         = [];
        for (var i = 0; i < this.settings.params.all.length; i++)
        {
            let buffer = this.vids[i];
            if (buffer instanceof GPUTexture)
            {
                entries.push({ binding: i, resource: buffer.createView() });
            }
            else
            {
                entries.push({ binding: i, resource: { buffer: buffer } });
            }
        }
        if (entries.length > 0)
        {
            const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: entries });
            this.bindGroup  = bindGroup;
        }
    };

    __createUniformBindGroup()
    {
        let device          = this.webCS.gpuDevice;
        let bindGroupLayout = this.uniformBindGroupLayout;
        let entries         = [];
        let nUniform        = Object.keys(this.settings.uniform).length;
        for (var i = 0; i < nUniform; i++)
        {
            let buffer = this.uniformVids[i];
            entries.push({ binding: i, resource: { buffer: buffer } });
        }
        if (entries.length > 0)
        {
            const bindGroup       = device.createBindGroup({ layout: bindGroupLayout, entries: entries });
            this.uniformBindGroup = bindGroup;
        }
    };
};

/**
 *WebCS hosts the adapter, gpu device, canvas, and creates CSKernel
 * */
class WebCS
{
    /*private*/ constructor(adapter, device, settings = {})
    {
        this.canvas = settings.canvas || document.createElement('canvas');
        if (settings.canvas == null)
        {
            let canvas    = document.createElement('canvas');
            canvas.width  = settings.width || 640;
            canvas.height = settings.height || 480;
            this.canvas   = canvas;
        }
        else
        {
            this.canvas = settings.canvas;
        }
        this.adapter       = adapter;
        this.gpuDevice     = device;
        this.SFmt2DataType = {};
        this.SFmt2Fmt      = {};
        this.Str2SFmt      = {};
        //this.__setFmt();
    };

    /*
     * Create WebCs object from adapter and device
     * @example
     *   let webCS = await WebCS.create({width:X, height:Y});
     * @example
     *   
     *    let webCS = await WebCS.create({ canvas: $('#canvas2GPU')[0] });
     */
    static async create(settings = {})
    {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter)
        {
            console.log('Failed to get GPU adapter.');
            return;
        }
        const device = await adapter.requestDevice();
        return new WebCS(adapter, device, settings);
    };

    createShaderFromString(source, settings = {})
    {
        //console.log(source)

        const shaderModule      = this.gpuDevice.createShaderModule({ code: source });
        settings.commandEncoder = this.gpuDevice.createCommandEncoder();
        this.commandEncoder     = settings.commandEncoder;
        let webCS               = this;
        return new CSKernel(webCS, shaderModule, settings);
    };
    createShaderFromFunction(func, settings = {})
    {
        let csmain_str        = func();
        let layout_str        = '';
        let comments          = /(\/\/.*)|(\/\*[\s\S]*?\*\/)/g;
        let csmain_nocomments = csmain_str.replace(comments, '');
        let global_str        = '';
        let global_func_str   = (this.glsl_functions || '') + '\n' + (settings.glsl_functions || '');
        var isWhite           = function(ch) {
            return ((ch == ' ') || (ch == '\t') || (ch == '\n'));
        };
        // process the shared
        if (true)
        {
            let re_shared = /shared\s+[^;]+;/g;
            let matches   = [...csmain_nocomments.matchAll(re_shared)];
            for (let match of matches)
            {
                global_str = global_str + match[0];
            }
            csmain_nocomments = csmain_nocomments.replace(re_shared, '');
        }

        // porcess the function
        if (true)
        {
            let func_si = csmain_nocomments.indexOf('function');
            if (func_si > 0)
            {
                function indexOfendf(str, si)
                {
                    let l      = str.length;
                    let ending = 0;
                    for (let iii = si; iii < l; iii++)
                    {
                        if (str[iii] == '{')
                            ending = ending + 1;
                        if (str[iii] == '}')
                        {
                            ending = ending - 1;
                            if (ending == 0)
                            {
                                return iii + 1;
                            }
                        }
                    }
                    return null;
                }
                while (func_si > 0)
                {
                    if (isWhite(csmain_nocomments[func_si + 8]))
                    {
                        let funcEndI = indexOfendf(csmain_nocomments, func_si + 8);
                        if (funcEndI == null)
                        {
                            // Error
                        }
                        global_func_str = global_func_str + '\n' +
                            'fn ' + csmain_nocomments.substring(func_si + 8, funcEndI);
                        csmain_nocomments =
                            csmain_nocomments.substring(0, func_si) + csmain_nocomments.substring(funcEndI);
                        func_si = csmain_nocomments.indexOf('function');
                    }
                    else
                    {
                        func_si = csmain_nocomments.indexOf('function', func_si + 8);
                    }
                }
            }
        }

        // process the module-scope const
        if (true)
        {
            let func_si = csmain_nocomments.indexOf('const');
            if (func_si > 0)
            {
                while (func_si > 0)
                {
                    if (isWhite(csmain_nocomments[func_si + 5]))
                    {
                        let funcEndI = csmain_nocomments.indexOf(';', func_si + 5) + 1;
                        if (funcEndI == null)
                        {
                            // Error
                        }
                        let myvar       = 'let ' + csmain_nocomments.substring(func_si + 5, funcEndI);
                        global_func_str = global_func_str + '\n' + myvar;
                        csmain_nocomments =
                            csmain_nocomments.substring(0, func_si) + csmain_nocomments.substring(funcEndI);
                        func_si = csmain_nocomments.indexOf('const');
                    }
                    else
                    {
                        func_si = csmain_nocomments.indexOf('const', func_si + 5);
                    }
                }
            }
        }

        // process the var<workgroup>
        if (true)
        {
            let func_si = csmain_nocomments.indexOf('var<workgroup>');
            if (func_si > 0)
            {
                while (func_si > 0)
                {
                    let funcEndI = csmain_nocomments.indexOf(';', func_si + 14) + 1;
                    if (funcEndI == null)
                    {
                        // Error
                    }
                    let myvar         = csmain_nocomments.substring(func_si, funcEndI);
                    global_func_str   = global_func_str + '\n' + myvar;
                    csmain_nocomments = csmain_nocomments.substring(0, func_si) + csmain_nocomments.substring(funcEndI);
                    func_si           = csmain_nocomments.indexOf('var<workgroup>');
                }
            }
        }

        // process the parameters
        if (true)
        {
            let func_str  = func.toString();
            let startI    = func_str.indexOf('(');
            let endI      = func_str.indexOf(')');
            let param_str = func_str.substring(startI + 1, endI).replace(/\s/g, '');
            let params    = param_str.split(',');
            if (settings.params == null)
            {
                settings.params = { all: params };
            }
            else
            {
                settings.params.all = params;
            }
            if (true)
            {
                let that = this;
                params.forEach(function(ele, idx) {
                    if (settings.params[ele] == null)
                    {
                        settings.params[ele] = { index: idx, type: that.__parsetype('buffer') };
                    }
                    else
                    {
                        settings.params[ele].index = idx;
                    }
                });
            }

            // procecss [], detect readonly/writeonly
            if (true)
            {
                let params_tex =
                    Object.keys(settings.params)
                        .filter(key => settings.params[key].type != null && settings.params[key].type.dim === 2);
                if (0 == params_tex.length)
                {
                }
                else
                {
                    // process the texture[]
                    if (true)
                    {
                        for (let texname of params_tex)
                        {
                            let texreader2 = new RegExp(
                                ['(', texname, ')', '\\s*\\[([^\\[\\]]+)\\]\s*\\[([^\\[\\]]+)\\]'].join(''), 'g');
                            let texwriter2 = new RegExp(
                                ['(', texname, ')', '\\s*\\[([^\\[\\]]+)\\]\\s*\\[([^\\[\\]]+)\\]\\s=([^;]+);'].join(
                                    ''),
                                'g');
                            let texreader = new RegExp(['(', texname, ')', '\\s*\\[([^\\[\\]]+)\\]'].join(''), 'g');
                            let texwriter =
                                new RegExp(['(', texname, ')', '\\s*\\[([^\\[\\]]+)\\]\\s*=([^;]+);'].join(''), 'g');
                            //let texreaders = [...csmain_nocomments.matchAll(texreader)];
                            //let texwriters = [...csmain_nocomments.matchAll(texwriter)];
                            csmain_nocomments = csmain_nocomments.replace(
                                texwriter2, 'textureStore($1,vec2<i32>(i32($3),i32($2)), $4);');
                            csmain_nocomments =
                                csmain_nocomments.replace(texwriter, 'textureStore($1,vec2<i32>($2), $3);');
                            csmain_nocomments =
                                csmain_nocomments.replace(texreader2, 'textureLoad($1,vec2<i32>(i32($3),i32($2)), 0);');
                            csmain_nocomments =
                                csmain_nocomments.replace(texreader, 'textureLoad($1,vec2<i32>($2), 0);');
                        }
                    }
                    // let's find out the imageStore
                    if (true)
                    {
                        let imgst_re = /textureStore\s*\(\s*([^,]+),/g;
                        let matches  = [...csmain_nocomments.matchAll(imgst_re)];
                        for (let match of matches)
                        {
                            var vname = match[1].trim();
                            if (settings.params[vname] == null)
                            {
                                // error
                            }
                            else
                            {
                                settings.params[vname].rwmode = 'w';
                            }
                        }
                    }
                }
                for (let paramname of settings.params.all)
                {
                    let memaccessor   = new RegExp(['[\\W](', paramname, ')', '\\s*\\['].join(''), 'g');
                    csmain_nocomments = csmain_nocomments.replace(memaccessor, ' $1.data[');
                }
            }

            // Declare the params in GLSL
            for (var pi = 0; pi < params.length; pi++)
            {
                let param_name = params[pi];
                let param_type = settings.params[param_name].type;
                if (param_type.dim == 1)
                { // buffer
                    let num_type = param_type.type;
                    // clang-format off
                     layout_str = layout_str +
                        ` struct struct_${param_name}{ data: array<${num_type}>;} ;\n@group(0) @binding(${pi}) var<storage, read_write> ${param_name} : struct_${param_name};\n`;
                    //            [[group(0), binding(2)]] var<storage, write> resultMatrix : array<f32>;
                    //layout (std430, binding = 0) buffer ssbA {  float A[]; };
                    //`layout (std430, binding = ${pi}) buffer ssb${param_name} {  ${num_type} ${param_name}[]; };`;
                    // clang-format on
                }
                else if (param_type.dim == 2)
                { // texture
                    // clang-format off
                    let pix_type = this.__str2sfmt(param_type);
                    let value_type = this.__sfmt2datatype(pix_type);

                    let rwmode = settings.params[param_name].rwmode == 'w' ?  'writeonly' : 'readonly';
                    let attr = settings.params[param_name].attr || "";
                    if((attr.indexOf('readonly') == -1 ) && (attr.indexOf('writeonly') == -1)){
                        attr = attr + " " + rwmode; 
                    }
                    if(attr.indexOf('readonly') > 0 ){
                        layout_str = layout_str +
                            `@group(0) @binding(${pi}) var ${param_name} : texture_2d<${value_type}>;\n`;
                    }else{
                        layout_str = layout_str +
                            `@group(0) @binding(${pi}) var ${param_name} : texture_storage_2d<${pix_type}, write>;\n`;
                    }
                    // clang-format on
                }
                else
                {
                    // error
                }
            }
        }
        // process the uniforms
        let unform_str = '';
        if (true)
        {
            settings.uniform = {};
            let re           = /this\.uniform\.([a-zA-Z0-9_-]{1,})\.([a-zA-Z0-9_-]{1,})/g;
            let re2          = /this\.uniform\.([a-zA-Z0-9_-]{1,})([^\.a-zA-Z0-9_-])+/g;
            let matches      = [...csmain_nocomments.matchAll(re)];
            let matches2     = [...csmain_nocomments.matchAll(re2)];
            let revar        = /[a-zA-Z0-9_-]{1,}/
            var indexOfSpace = function(s, startIndex) {
                let si = startIndex;
                while (!isWhite(s[si]))
                    si++;
                return si;
            };
            var indexOfNonSpace = function(s, startIndex) {
                let si = startIndex;
                while (isWhite(s[si]))
                    si++;
                return si;
            };
            var types = {
                'u32': 'vec4<u32>',
                'f32': 'vec4<f32>',
                'i32': 'vec4<i32>',
                'f64': 'vec4<f64>',
                'vec4<i32>': 'vec4<i32>',
                'vec4<u32>': 'vec4<u32>',
                'vec4<f32>': 'vec4<f32>',
                'vec4<f64>': 'vec4<f64>'
            };
            for (let match of matches)
            {
                var vname = match[1];
                if (settings.uniform[vname] == null)
                {
                    settings.uniform[vname] = { type: null, fields: {} }
                }
                settings.uniform[vname][match[2]] = 1;
                if (true)
                {
                    let lineStartI = 0;
                    lineStartI     = Math.max(lineStartI, csmain_nocomments.lastIndexOf(';', match.index));
                    lineStartI     = Math.max(lineStartI, csmain_nocomments.lastIndexOf('}', match.index));
                    lineStartI     = Math.max(lineStartI, csmain_nocomments.lastIndexOf('{', match.index));
                    lineStartI     = lineStartI + 1;
                    let type_si    = indexOfNonSpace(csmain_nocomments, lineStartI);
                    let type_ei    = indexOfSpace(csmain_nocomments, type_si);
                    let type_str   = csmain_nocomments.substring(type_si, type_ei);
                    let mytype     = types[type_str] || 'vec4<u32>';
                    settings.uniform[vname]['type'] = mytype;
                }
            }
            for (let match of matches2)
            {
                var vname = match[1];
                if (settings.uniform[vname] == null)
                {
                    settings.uniform[vname] = { type: null, fields: {} }
                }
            }
            let pi = 0;
            for (let uniform in settings.uniform)
            {
                let mytype                      = settings.uniform[uniform].type;
                settings.uniform[uniform].index = pi;
                // [[binding(0), group(0)]] var<uniform> params : SimParams;
                let my_uniform_str = `
                struct struct_${uniform} {data:${mytype};};
                @group(1) @binding(${pi}) var<uniform> ${uniform} : struct_${uniform};`;
                unform_str = unform_str + my_uniform_str;
            }
            if (true)
            {
                for (let uniform in settings.uniform)
                {
                    let memaccessor   = new RegExp(['this.uniform.(', uniform, ')', '\\.'].join(''), 'g');
                    csmain_nocomments = csmain_nocomments.replace(memaccessor, '$1.data.');
                }
                //csmain_nocomments =
                //    csmain_nocomments.replace(/this\.uniform\./g, '');
            }
        }

        let local_size = settings.local_size || [8, 8, 1];
        // clang-format off
        let source = `
        ${layout_str} 
        ${unform_str}
        ${global_str}
        const LOCAL_SIZE_X:u32 = ${local_size[0]}u;
        const LOCAL_SIZE_Y:u32 = ${local_size[1]}u;
        const LOCAL_SIZE_Z:u32 = ${local_size[2]}u;
        var<private> num_workgroups:vec3<u32>;
        var<private> workgroup_id:vec3<u32>;
        ${global_func_str}
        fn csmain(thread: vec3<u32>, localthread:vec3<u32>, workgroup_id:vec3<u32>){
            ${csmain_nocomments}
        }
        @compute @workgroup_size(${local_size[0]}, ${local_size[1]}, ${local_size[2]})

        fn main(@builtin(global_invocation_id) thread: vec3<u32>, @builtin(local_invocation_id) localthread: vec3<u32>, @builtin(workgroup_id) block: vec3<u32>, @builtin(num_workgroups) wgs:vec3<u32>) {
            num_workgroups = wgs;
            workgroup_id = block;
            csmain(thread, localthread, block);
        }
        `
        // clang-format on
        return this.createShaderFromString(source, settings);
    };

    __parsetype(type)
    {
        type    = type.replace(/\s*/g, '');
        let bi  = -1;
        let t   = 'f32';
        let dim = 1;
        if (type == 'buffer')
        {
            t   = 'f32';
            dim = 1;
        }
        else if (type == 'texture')
        {
            t   = 'rgba8unorm';
            dim = 2;
        }
        else if ((bi = type.indexOf('[][]')) >= 0)
        {
            t   = (bi == 0) ? 'rgba8unorm' : type.substring(0, bi).toLowerCase();
            dim = 2;
        }
        else if ((bi = type.indexOf('[]')) >= 0)
        {
            t = (bi == 0) ? 'f32' : type.substring(0, bi).toLowerCase();
            if (['f32', 'u32', 'i32'].includes(t))
            {
            }
            else if (t == 'float')
            {
                t = 'f32';
            }
            else
            {
                throw ('type is not supported : ' + t);
            }
            dim = 1;
        }
        else
        {
            // error
        }
        return { 'type': t, 'dim': dim };
    };

    /**
     *Create CSKernel from 'source code' or function 
     *@param {source_or_function} source - 'source code' or function
     *@param {{}} settings - input settings
     *@example
     *   let cs_smm = webCS.createShader(gpu_smm_naive, {local_size:[8, 8, 1], groups:[M/8, N/8, 1]});
     */
    createShader(source, settings = {})
    {
        if (true)
        {
            // convert settings.params
            if (settings.params != null)
            {
                for (let key in settings.params)
                {
                    let v = settings.params[key];
                    if (_isString(v))
                    {
                        settings.params[key] = { type: this.__parsetype(v) };
                    }
                }
            }
        }
        if (typeof source === 'string')
        {
            return this.createShaderFromString(source, settings);
        }
        else
        {
            return this.createShaderFromFunction(source, settings);
        }
    };

    addFunctions(func)
    {
        this.glsl_functions = this.glsl_functions || '';
        this.glsl_functions += func;
    };

    /**
     * Present the tex to this.canvas 
     * @param {GPUTexture} tex - texture to be presented
     */
    async present(tex)
    {
        console.log('present');
        const canvas             = this.canvas;
        const context            = this.canvas.getContext('webgpu');
        const presentationFormat = context.getPreferredFormat(this.adapter);
        let device               = this.gpuDevice;
        const presentationSize   = [
            canvas.width,
            canvas.height,
        ];

        /*
        if (tex == undefined)
        {
            tex       = this.createTexture('rgba8unorm', canvas.width, canvas.height);
            let image = document.getElementById('image000');
            await image.decode();
            let w           = canvas.width;
            let h           = canvas.height;
            let imageBitmap = await createImageBitmap(image);
            device.queue.copyExternalImageToTexture(
                { source: imageBitmap }, { texture: tex }, [imageBitmap.width, imageBitmap.height]);
        }
        */
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
        const fullscreenQuadPipeline     = device.createRenderPipeline({
            vertex: {
                module: device.createShaderModule({
                    code: fullscreenTexturedQuadWGSL,
                }),
                entryPoint: 'vert_main',
            },
            fragment: {
                module: device.createShaderModule({
                    code: fullscreenTexturedQuadWGSL,
                }),
                entryPoint: 'frag_main',
                targets: [
                    {
                        format: presentationFormat,
                    },
                ],
            },
            primitive: {
                topology: 'triangle-strip',
            },
        });
        const sampler                    = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });
        const renderBindGroup            = device.createBindGroup({
            layout: fullscreenQuadPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: sampler,
                },
                {
                    binding: 1,
                    resource: tex.createView(),
                },
            ],
        });
        let commandEncoder               = device.createCommandEncoder();
        const passEncoder                = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: context.getCurrentTexture().createView(),
                    clearValue: { r: 0.0, g: 1.0, b: 0.0, a: 1.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
        });

        passEncoder.setPipeline(fullscreenQuadPipeline);
        passEncoder.setBindGroup(0, renderBindGroup);
        passEncoder.draw(4, 1, 0, 0);
        passEncoder.end();
        device.queue.submit([commandEncoder.finish()]);
    };
    createBuffer(size)
    {
        let device     = this.gpuDevice;
        let gpuBuffer  = device.createBuffer({
            mappedAtCreation: true,
            size: size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        gpuBuffer.size = size;
        return gpuBuffer;
    };
    createTexture(fmt, w, h)
    {
        fmt = fmt || 'rgba8unorm';
        w   = w || this.canvas.width;
        h   = h || this.canvas.height;

        let device = this.gpuDevice;
        let tex    = device.createTexture({
            size: {
                width: w,
                height: h,
            },
            format: fmt,
            usage: //GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });
        tex.size   = [w, h, 4];
        return tex;
    };
    /**
     *Copy for gpu vid to host arrary
     */
    async getData(vid, dstarray)
    {
        function isTypedArray(arr)
        {
            return ArrayBuffer.isView(arr) && !(arr instanceof DataView);
        }
        function getType(arr)
        {
            return isTypedArray(arr) && arr.constructor.name;
        }
        function getHostAccessArrary(dstarray, arrayBuffer)
        {
            let hostAccessArrary = null;
            if (dstarray == undefined)
            {
                hostAccessArrary = new Uint8Array(arrayBuffer);
            }
            else if (typeof dstarray === 'string')
            {
                const typemap = new Map([
                    ['int8', Int8Array], ['uint8', Uint8Array], ['uint8clamped', Uint8ClampedArray],
                    ['int16', Int16Array], ['uint16', Uint16Array], ['int32', Int32Array], ['uint32', Uint32Array],
                    ['float32', Float32Array], ['float', Float32Array], ['float64', Float64Array],
                    ['bigint', BigInt64Array], ['bigint', BigUint64Array]
                ]);
                if (typemap.has(dstarray))
                {
                    let obj          = typemap.get(dstarray);
                    hostAccessArrary = new obj(arrayBuffer);
                }
                else
                {
                    hostAccessArrary = new Uint8Array(arrayBuffer);
                }
            }
            else if (isTypedArray(dstarray))
            {
                hostAccessArrary = new dstarray.constructor(arrayBuffer);
            }
            else
            {
                hostAccessArrary = new Uint8Array(arrayBuffer);
            }
            return hostAccessArrary;
        }

        const resultBufferSizeInBytes = vid.size || 64 * 64 * 4;

        const gpuReadBuffer = this.gpuDevice.createBuffer(
            { size: resultBufferSizeInBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        //let commandEncoder = this.commandEncoder; // This make the 2nd call fail
        let commandEncoder = this.gpuDevice.createCommandEncoder();
        // Encode commands for copying buffer to buffer.
        commandEncoder.copyBufferToBuffer(
            vid /* source buffer */, 0 /* source offset */, gpuReadBuffer /* destination buffer */,
            0 /* destination offset */, resultBufferSizeInBytes /* size */
        );

        // Submit GPU commands.
        const gpuCommands = commandEncoder.finish();
        this.gpuDevice.queue.submit([gpuCommands]);

        // Read buffer.
        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer    = gpuReadBuffer.getMappedRange();
        let hostAccessArrary = getHostAccessArrary(dstarray, arrayBuffer);

        if (isTypedArray(dstarray) && (dstarray.length == hostAccessArrary.length))
        {
            dstarray.set(hostAccessArrary);
        }
        else
        {
            dstarray = hostAccessArrary.slice();
        }
        gpuReadBuffer.unmap();
        return dstarray;
    }

    __setFmt()
    {
        // clang-format off
        let fmts = [
            // SFmt,       Fmt  ,        DataType, str
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
        ];
        // clang-format on
        for (let fmt of fmts)
        {
            this.SFmt2DataType[fmt[0]] = fmt[2];
            this.SFmt2Fmt[fmt[0]]      = fmt[1];
            this.Str2SFmt[fmt[3]]      = fmt[0];
        }
    };
    __sfmt2datatype(fmt)
    {
        return this.SFmt2DataType[fmt] || 'f32';
    };
    __sfmt2fmt(fmt)
    {
        return this.SFmt2Fmt[fmt] || 'rgba8unorm';
    };
    __str2sfmt(str)
    {
        return this.Str2SFmt[str] || 'rgba8unorm';
    };
};
win.WebCS = WebCS;
})(window);
