import './matrix.css'

import {App} from '/js/webCS.html.js'

import {WebCS} from '../src/webcs.js'

import {displayMatrix} from './matrix.js'

let webCS        = null;
let cs_smm_naive = null;
let cs_texture   = null;
let cs_img_dwt   = null;
let cs_kernels   = {};
let gpu_kernels  = {};
let do_cs        = {};
var X = 512, Y = 512, Z = 1;
(function() {
let testcases = ['smm_naive', 'texture', 'texture2', 'img_texture', 'img_dwt', 'histogram', 'filter', 'filter2', 'save_texture'];
// clang-format off
function gpu_smm_naive(A,B,C){
           return `
               // It is optional to decalare the src or dst in wgsl.
               // Another option is to decalre them when  webCS.createShader : 
               //     param:{A:"f32[]", B:"f32[]", C:"f32[]"}
               var A:array<f32>;
               // var B:array<f32>;
               // var C:array<f32>;

               //  declare the uniform,  must use 'this.uniform'
               // C[M, N] = A[M, K] * B[K, N]
               var mnk:vec4u = this.uniform.MNK;
               var M:u32 = mnk.x;
               var N:u32 = mnk.y;
               var K:u32 = mnk.z;
               // Compute a single element C[thread.y, thread.x] by looping over k
               var sum:f32 = 0.0;
               for (var k:u32 = 0u; k < K; k = k+1u)
               {
                   sum = sum + A[thread.y * K + k] * B[k * N + thread.x];
               }
        
               // Store the result
               C[thread.y*N + thread.x] = sum;
           `;
}

function gpu_texcopy(src, dst){
    return `
        dst[thread.y][thread.x] = src[thread.y][thread.x];     
        `;
}
function gpu_texture2(src, dst){
    return `
        var pos:vec2<u32> = vec2<u32>(thread.xy);
        // vec4 pixel = imageLoad(src, pos); or vec4 pixel = src[pos]
        var pixel:vec4<f32> = src[pos.y][pos.x]; 
        var invert:vec4<f32> = vec4<f32>(1.0 - pixel.x, 1.0 - pixel.y, 1.0 - pixel.z, 1.0);
        // imageStore(dst, pos, invert); or dst[pos] = invert;
        dst[pos.y][pos.x] = invert;     
    `;
}
function gpu_texture(dst){
        return `
        var pos:vec2<u32> = vec2<u32>(thread.xy);
        var x : f32 = f32(thread.x);
        var y : f32 = f32(thread.y);
        //imageStore(dst, pos, vec4(x / (y+1.0+x), y / (y+1.0+x),  0.0, 1.0));
        dst[thread.y][thread.x] =  vec4<f32>(x / (y+1.0+x), y / (y+1.0+x),  0.0, 1.0);
        `;
}
function gpu_img_dwt(src, dst){
    return `
        function  YSize() -> u32{ return u32(g_workgroup_id.y*g_num_workgroups.y);}
        function  XSize() -> u32{ return u32(g_workgroup_id.x*g_num_workgroups.x);} 
        var x:u32 = u32(thread.x);
        var y:u32 = u32(thread.y);
        var p00: vec4<f32> = src[2u*y + 0u][2u*x + 0u];
        var p01: vec4<f32> = src[2u*y + 0u][2u*x + 1u];
        var p10: vec4<f32> = src[2u*y + 1u][2u*x + 0u];
        var p11: vec4<f32> = src[2u*y + 1u][2u*x + 1u];
        dst[y][x] = (p00 + p01 + p10 + p11) / 4.0;
        dst[y][x + XSize()] = (p00 + p10 - p01 - p11) / 4.0;
        dst[y + YSize()][x] = (p00 + p01 - p10 - p11) / 4.0;
        dst[y + YSize()][x + XSize()] = (p00 + p11 - p01 - p10) / 4.0;
        `;
}
function gpu_histogram(src, dst){
        return `
        var pixel:vec4<f32> = src[thread.xy]; 
        var gray: f32 = 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
        var grayu:u32 = u32(floor(gray * 255.0));
        grayu = grayu & 255u;
        dst[grayu] = dst[grayu] + 1.0;
        //int ru = int(floor(pixel.r*255.0));
        //int gu = int(floor(pixel.g*255.0));
        //int bu = int(floor(pixel.b*255.0));
        //dst[ru + 256] = dst[ru+256] + 1.0;
        //dst[gu + 256*2] = dst[gu+256*2] + 1.0;
        //dst[bu + 256*3] = dst[bu+256*3] + 1.0;
`;
}
function gpu_replace(src, dst, histo){
        return `
        var pixel : vec4<f32> = src[thread.xy]; 
        var gray:f32 = 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
        var grayu:u32 = u32(floor(gray * 255.0));
        grayu = grayu & 255u;
        var new_gray:f32 = histo[grayu];
        var diff_gray:f32 = new_gray - gray;
        var new_pixel:vec4<f32> = vec4(pixel.r + diff_gray * 0.2126, pixel.g + diff_gray * 0.7152, pixel.b + diff_gray * 0.0722, 1.0);
        dst[thread.xy] = new_pixel;
        `;
}

function gpu_filter(src, dst){
	return `
    const kernel = mat3x3f(
        1.0,1.0,1.0,
        0.0,0.0,0.0,
        -1.0,-1.0,-1.0);
    var pos:vec2<u32> = vec2<u32>(thread.xy);
    var sum:vec4<f32> = vec4<f32>(0.0,0.0,0.0,1.0);
    for(var j:u32=0; j<3; j++){
        for(var i:u32=0; i<3; i++){
            let pixel = src[pos.y + j -1][pos.x + i -1];
            sum = sum + pixel * kernel[j][i];
        }
    }
    dst[pos.y][pos.x] = sum;     
    `;
}

function gpu_filter2(src, dst){
	return `
    // It is optional to decalare the src or dst in wgsl.
    // Another option is to decalre them when  webCS.createShader : 
    //     params: { src: 'texture', 'dst': 'texture' }  // compiler will deduce the final type 
    var src : texture_2d<f32>;
    //var dst:texture_storage_2d<rgba8unorm,write>;
    var kernel:mat3x3f = this.uniform.KERNEL;
    var pos:vec2<u32> = vec2<u32>(thread.xy);
    var sum:vec4<f32> = vec4<f32>(0.0,0.0,0.0,1.0);
    for(var j:u32=0; j<3; j++){
        for(var i:u32=0; i<3; i++){
            let pixel = src[pos.y + j -1][pos.x + i -1];
            sum = sum + pixel * kernel[j][i];
        }
    }
    dst[pos.y][pos.x] = sum;     
    `;
}
// clang-format on
gpu_kernels.smm_naive   = gpu_smm_naive;
gpu_kernels.texture     = gpu_texture;
gpu_kernels.texture2    = gpu_texture2;
gpu_kernels.img_texture = gpu_texture2;
gpu_kernels.img_dwt     = gpu_img_dwt;
gpu_kernels.histogram   = gpu_histogram;
gpu_kernels.filter      = gpu_filter;
gpu_kernels.filter2     = gpu_filter2;
gpu_kernels.save_texture= gpu_texture2;

// menus for example
(function() {
do_cs.do_smm_naive = async function(kernel_name) {
    // cpuC = cpuA * cpuB
    var M = 64, N = 64, K = 64;
    var createArray = function(n) {
        var buf = new Float32Array(n);
        for (var i = 0; i < n; i++)
        {
            buf[i] = Math.random();
        }
        return buf;
    };
    let cpuA = createArray(M * K);
    let cpuB = createArray(K * N);
    let cpuC = createArray(M * N);
    //if (cs_smm_naive == null)
    {
        cs_smm_naive = webCS.createShader(gpu_smm_naive, { local_size: [8, 8, 1], groups: [M / 8, N / 8, 1] });
    }

    const t0 = performance.now();
    await cs_smm_naive.run(cpuA, cpuB, cpuC, { 'MNK': [M, N, K, 0] });
    const t1 = performance.now();
    let t    = t1 - t0;
    $('#time').html(t.toFixed(1).toString());
    if (true) // Check result
    {
        let acc = 0, x = Math.floor(N * Math.random()), y = Math.floor(N * Math.random());
        for (let k = 0; k < K; k++)
            acc += cpuA[y * K + k] * cpuB[k * N + x];
        cpuC       = await cs_smm_naive.getData('C', 'float');
        let result = cpuC[y * N + x];
        let diff   = result - acc;
        result     = result.toFixed(7);
        diff       = diff.toFixed(7);
        App.showMessage(`Cgpu[${y},${x}] = ${result}; \u2003<br/> Cgpu[${y},${x}] - Ccpu[${y},${x}] : ${diff}`);
        displayMatrix(cpuA, 'gpuA', $('#data0_div')[0], M, K);
        displayMatrix(cpuB, 'gpuB', $('#data1_div')[0], K, K);
        displayMatrix(cpuC, 'gpuC', $('#data2_div')[0], M, N);
        $('#data_div').show();
    }
    $('#code_smm_naive').show();
};
do_cs.do_texture = async function(kernel_name) {
    //ivec2 storePos = ivec2(gl_GlobalInvocationID.xy);
    //imageStore(dst, storePos, vec4(vec2(gl_WorkGroupID.xy) / vec2(gl_NumWorkGroups.xy), 0.0, 1.0));
    //if (cs_texture == null)
    {
        cs_texture = webCS.createShader(
            gpu_texture, { local_size: [8, 8, 1], groups: [X / 8, Y / 8, 1], params: { 'dst': 'texture' } });
    }

    await cs_texture.run(null);

    let tex = cs_texture.getTexture('dst');
    webCS.present(tex);
    $('#display1')[0].appendChild(webCS.canvas);
    $(webCS.canvas).show();
};

do_cs.do_texture2 = async function(kernel_name) {
    do_cs.do_texture();
    //if (cs_kernels['texture2'] == null)
    {
        cs_kernels['texture2'] = webCS.createShader(
            gpu_texture2,
            { local_size: [8, 8, 1], groups: [X / 8, Y / 8, 1], params: { src: '[][]', 'dst': 'rgba8[][]' } });
    }

    let texSrc = cs_texture.getTexture('dst');
    await cs_kernels['texture2'].run(texSrc, null);

    let tex = cs_kernels['texture2'].getTexture('dst');
    webCS.present(tex);
    $('#display1')[0].appendChild(webCS.canvas);
    $(webCS.canvas).show();
};

do_cs.do_img_dwt = async function(kernel_name) {
    //if (cs_kernels.cs_img_dwt == null)
    {
        cs_kernels.cs_img_dwt = webCS.createShader(
            gpu_img_dwt,
            { local_size: [8, 8, 1], groups: [X / 16, Y / 16, 1], params: { src: 'texture', 'dst': 'texture' } });
    }
    //if (cs_kernels.cs_texcopy == null)
    {
        cs_kernels.cs_texcopy = webCS.createShader(
            gpu_texcopy, { local_size: [8, 8, 1], groups: [X / 8, Y / 8, 1], params: { src: '[][]', 'dst': '[][]' } });
    }

    let texSrc  = $('#image000')[0];
    let texDst1 = webCS.createTexture();
    let texDst2 = webCS.createTexture();
    await (cs_kernels.cs_img_dwt.setGroups(X / 16, Y / 16, 1)).run(texSrc, texDst1);
    // copy the LL0 to texDst1
    await (cs_kernels.cs_texcopy.setGroups(X / 16, Y / 16, 1)).run(texDst1, texDst2);
    // DWT the LL0
    await cs_kernels.cs_img_dwt.run(texDst2, texDst1, X / 32, Y / 32, 1);

    let tex = cs_kernels.cs_img_dwt.getTexture('dst');
    webCS.present(tex);
    $('#display1')[0].appendChild(webCS.canvas);
    $(webCS.canvas).show();
};

do_cs.do_img_texture = async function(kernel_name) {
    //if (cs_kernels['texture2'] == null)
    {
        cs_kernels['texture2'] = webCS.createShader(
            gpu_texture2,
            { local_size: [8, 8, 1], groups: [X / 8, Y / 8, 1], params: { src: 'texture', 'dst': 'texture' } });
    }

    let texSrc = $('#image000')[0];
    await cs_kernels['texture2'].run(texSrc, null);

    let tex = cs_kernels['texture2'].getTexture('dst');
    webCS.present(tex);
    $('#display1')[0].appendChild(webCS.canvas);
    $(webCS.canvas).show();
};

do_cs.do_save_texture = async function(kernel_name) {
    //if (cs_kernels['texture2'] == null)
    {
        cs_kernels['texture2'] = webCS.createShader(
            gpu_texture2,
            { local_size: [8, 8, 1], groups: [X / 8, Y / 8, 1], params: { src: 'texture', 'dst': 'texture' } });
    }

    let texSrc = $('#image000')[0];
    await cs_kernels['texture2'].run(texSrc, null);

    let tex = cs_kernels['texture2'].getTexture('dst');
    webCS.present(tex);
    let jpeg = webCS.canvas.toDataURL("image/jpeg", 0.5);
    //$('#display1')[0].appendChild(webCS.canvas);
    function addSaveBtn (box, dataurl)
    {
        let savebtn =  box.find("#save");
        if(savebtn.length == 0)
        {
            box.append('<div><a href="#" id="save"  class="test_case" download="image.png">Download</a></div>'); 
            savebtn =  box.find("#save");
            savebtn.click(function(){
                console.log(dataurl);
                savebtn[0].href = dataurl;
            });
        }
        savebtn.show();
    }
    function presentImg(dataurl){
        let image2 = $('#display1').find("#image2GPU");
        if(image2.length == 0){
            image2 = $("<img id='image2GPU' >");
            $('#display1').append(image2);
            $('#display1').append("<div></div>");
        }
        image2[0].src=jpeg;
        image2.show();
    }
    addSaveBtn( $('#display1'), jpeg);
    presentImg(jpeg);
};

do_cs.do_filter = async function(kernel_name) {
    //if (cs_kernels['filter'] == null)
    {
        cs_kernels['filter'] = webCS.createShader(
            gpu_filter,
            { local_size: [8, 8, 1], groups: [X / 8, Y / 8, 1], params: { src: 'texture', 'dst': 'texture' } });
    }

    let texSrc = $('#image000')[0];
    await cs_kernels['filter'].run(texSrc, null);

    let tex = cs_kernels['filter'].getTexture('dst');
    webCS.present(tex);
    $('#display1')[0].appendChild(webCS.canvas);
    $(webCS.canvas).show();
};

do_cs.do_filter2 = async function(kernel_name) {
    //if (cs_kernels['filter2'] == null)
    {
        cs_kernels['filter2'] = webCS.createShader(
            gpu_filter2,
            { local_size: [8, 8, 1], groups: [X / 8, Y / 8, 1], params: { src: 'texture', 'dst': 'texture' } });
    }

    let texSrc = $('#image000')[0];
    // mat3x3f AlignOf(vec) == 16(4float), so each vector has 16 bytes in memory
    await cs_kernels['filter2'].run(texSrc, null, {'KERNEL':[1.0,1.0,1.0,0.0,   0.0,0.0,0.0,0.0,   -1.0,-1.0,-1.0,0.0]});
    let tex = cs_kernels['filter2'].getTexture('dst');
    webCS.present(tex);
    $('#display1')[0].appendChild(webCS.canvas);
    $(webCS.canvas).show();
};


do_cs.do_general = async function(kernel_name) {
    //    if (cs_kernels['texture2'] == null)
    {
        cs_kernels['texture2'] = webCS.createShader(
            gpu_kernels[kernel_name],
            { local_size: [8, 8, 1], groups: [X / 8, Y / 8, 1], params: { src: 'texture', 'dst': 'texture' } });
    }

    let texSrc = $('#image000')[0];
    await cs_kernels['texture2'].run(texSrc, null);

    let tex = cs_kernels['texture2'].getTexture('dst');
    webCS.present(tex);
    $('#display1')[0].appendChild(webCS.canvas);
    $(webCS.canvas).show();
};


do_cs.do_histogram = async function(kernel_name) {
    // 1. create histogram kernel
    if (cs_kernels['histogram '] == null)
    {
        cs_kernels['histogram '] = webCS.createShader(
            gpu_histogram,
            { local_size: [8, 8, 1], groups: [X / 8, Y / 8, 1], params: { src: 'texture', 'dst': 'float[]' } });
    }
    let texSrc = $('#image000')[0];
    let bufHis = webCS.createBuffer(256 * 4 * 4);
    // 2. run histogram kernel
    await cs_kernels['histogram '].run(texSrc, bufHis);

    // 3. get new color based on histogram
    let histogram = await cs_kernels['histogram '].getData('dst', 'float');
    for (let ii = 1; ii < 256; ii++)
    {
        histogram[ii] = histogram[ii - 1] + histogram[ii];
    }
    let cdf0 = 0, cdf1 = histogram[255];
    let newColor = new Float32Array(256);
    for (let ii = 0; ii < 256; ii++)
    {
        if (histogram[ii] == 0)
        {
            continue;
        }
        if (cdf0 == 0)
        {
            cdf0 = histogram[ii];
        }
        newColor[ii] = (histogram[ii] - cdf0) / (cdf1 - cdf0) * 255.0 / 255.0;
    }
    console.log(histogram);

    // 4. create replace kernel
    if (cs_kernels['replace'] == null)
    {
        cs_kernels['replace'] = webCS.createShader(gpu_replace, {
            local_size: [8, 8, 1],
            groups: [X / 8, Y / 8, 1],
            params: { src: 'texture', 'dst': 'texture', histo: 'float[]' }
        });
    }
    // 5. run replace  kernel with newColor
    await cs_kernels['replace'].run(texSrc, null, newColor);
    let tex = cs_kernels['replace'].getTexture('dst');
    webCS.present(tex);
    $('#display1')[0].appendChild(webCS.canvas);
    $('#canvas2GPU').show();
};

function doExampleGPU(e, ui)
{
    let myfilter = ui.item.attr('data-filter');
    $('.code.example').hide();
    $('#canvas2GPU').hide();
    $('.test_case').hide();
    $('#image2GPU').hide();
    $('#data_div').hide();
    $('#code_' + myfilter).show();
    if( 'do_' + myfilter in do_cs){
        do_cs['do_' + myfilter](myfilter);
    } else{
        do_cs['do_general'](myfilter);
    }
}
async function doExample(e, ui)
{
    let myfilter = ui.item.attr('data-filter');
    if (webCS == null)
    {
        webCS = await WebCS.create({ canvas: $('#canvas2GPU')[0] });
        //webCS = new WebCS();
    }
    if (myfilter == null)
    {
        return;
    }
    disablePractice();
    doExampleGPU(e, ui);
}

function disablePractice()
{
    $('#GPUToggleButton').prop('checked', true);
    $('#GPUToggleButton').click();
    $('#practice_div').hide();
    $('#code_div').show();
}

// append menus
(function appendmenu() {
    function doone(thefilters, themenu)
    {
        let ula = $('<ul/>');
        thefilters.forEach(function(ele) {
            ula.append('<li data-filter=\'' + ele + '\'><div>' + ele + '</div></li>');
        });
        $(themenu).append(ula.children().detach());
    }
    doone(testcases, '#example_menus');
    doone(testcases, '#loadexample_menus');
})();


$('#examplemenu').menu({ select: doExample });
})();

(async function setupExample() {
    /*
	let example_js       = {};
	example_js.smm_naive = `
        let M = 64, N = 64, K =64;
        var createArray = function ( n) { 
            var buf = new Float32Array(n);
            for(var i = 0; i < n; i++){buf[i] = Math.random();}
            return buf;
        };
        let cpuA = createArray(M*K);
        let cpuB = createArray(K*N);
        let cpuC = createArray(M*N);
        let webCS = await WebCS.create();
        let cs_smm = webCS.createShader(gpu_smm_naive, {local_size:[8, 8, 1], groups:[M/8, N/8, 1]});
        await (cs_smm.setUniform('MNK', M, N, K, 0)).run(cpuA, cpuB, cpuC);
        // or
        // cs_smm_naive.run(cpuA, cpuB, cpuC, {'MNK':[M,N,K,0]});
        cs_smm.getData('C', cpuC);
            `;

	example_js.texture = `
        var X = 512, Y = 512, Z = 1;
        let webCS = await WebCS.create({width:X, height:Y});
        // or let webCS = new WebCS({canvas:$("#canvas2GPU")[0]});
        let cs_texture = webCS.createShader(gpu_texture, { local_size:[8, 8, 1], groups:[X/8, Y/8, 1], params:{'dst':'texture'}});

        await cs_texture.run(null);

        let tex = cs_texture.getTexture('dst');
        webCS.present(tex);
        $("#display1")[0].appendChild(webCS.canvas);
        `;

	example_js.texture2 = `
        cs_texture().run(null); // to gerenate a texture
        var X = 512, Y = 512, Z = 1;
        let webCS = await WebCS.create({width:X, height:Y});
        // or let webCS = new WebCS({canvas:$("#canvas2GPU")[0]});
        let cs_texture2 = webCS.createShader(gpu_texture2, { local_size:[8, 8, 1], groups:[X/8, Y/8, 1], params:{src:'texture', 'dst':'texture'}});

        let texSrc = cs_texture.getTexture('dst');
        await cs_texture2.run(texSrc, null);

        let tex = cs_texture2.getTexture('dst');
        webCS.present(tex);
        $("#display1")[0].appendChild(webCS.canvas);
        `;

	example_js.img_texture = `
        var X = 512, Y = 512, Z = 1;
        let webCS = await WebCS.create({width:X, height:Y});
        // or let webCS = new WebCS({canvas:$("#canvas2GPU")[0]});
        let cs_texture2 = webCS.createShader(gpu_texture2, { local_size:[8, 8, 1], groups:[X/8, Y/8, 1], params:{src:'texture', 'dst':'texture'}});

        let texSrc = $('#image000')[0];
        await cs_texture2.run(texSrc, null);

        let tex = cs_texture2.getTexture('dst');
        webCS.present(tex);
        $("#display1")[0].appendChild(webCS.canvas);
        `;
    */
    //$("#code_smm_naive").hide();
    let codes_block = $('#codes_block');
    testcases.forEach(function(ele) {
        let code_ele  = $(`<code id='code_${ele}' class='code example language-javascript'></code>`)
        var btf = hljs.highlight('c++', gpu_kernels[ele].toString());
        let btf_value = btf.value;
        if (ele == 'histogram')
        {
            let r2    = hljs.highlight('c++', gpu_replace.toString()).value;
            btf_value = btf_value + '\n' + r2;
        }
        const use_example = false;
        let btf2          = hljs.highlight(
            'javascript',
            /*example_js[ele] ||*/ (('do_' + ele in do_cs) ? do_cs['do_' + ele] : do_cs['do_general']).toString());
        let btf2_value    = btf2.value.replace('gpu_' + ele, '<span class="hljs-title">gpu_' + ele + '</span>');
        if (use_example)
        {
            btf2_value = 'await (async function(kernel_name) {' + btf2_value + '\n})();';
        }
        else
        {
            btf2_value = 'await (' + btf2_value + ')();';
        }
        code_ele.html(
            '<div class="gpu_code example">' + btf_value + '<br/><br/>        ' +
            '</div>' +
            '<div class="js_code example">' + btf2_value + '</div>');
        codes_block.append(code_ele);
    });
    $('.code.example').hide();
    $('#code_smm_naive').show();
})();
})();

$(function() {
    var editorgpu = CodeMirror.fromTextArea(
        document.getElementById('practisegpu'), { lineNumbers: true, mode: 'text/x-c++src', matchBrackets: true });
    var editorjs = CodeMirror.fromTextArea(
        document.getElementById('practisejs'), { lineNumbers: true, mode: 'javascript', matchBrackets: true });
    $('#run_practise').button();
    $('#run_practise').click(function() {
        formatall(editorgpu);
        formatall(editorjs);
        let test_str =
            //editorgpu.getValue() + '\n async function mypractice(){' + editorjs.getValue() + '}; mypractice();';
            editorgpu.getValue() + '\n (' + editorjs.getValue() + ')("test")'; 
        eval(test_str);
    });

    function formatall(editor)
    {
        let oldSize    = editorgpu.getScrollInfo()
        var totalLines = editor.lineCount();
        editor.autoFormatRange({ line: 0, ch: 0 }, { line: totalLines });
        editorgpu.setSize(oldSize.width, oldSize.height);
    }
    function setPractice()
    {
        if ($('#GPUToggleButton').is(':checked'))
        {
            $('#practice_div').show();
            $('#code_div').hide();
        }
        else
        {
            $('#practice_div').hide();
            $('#code_div').show();
        }

    }
    if (document.location.hash === '#practise')
    {
        $('#GPUToggleButton').prop('checked', true);
    }
    setPractice();
    $('#GPUToggleButton').click(setPractice);

    // setup load
    async function loadExample(e, ui)
    {
        let myfilter = ui.item.attr('data-filter');
        if (myfilter == null)
        {
            return;
        }
        $('#GPUToggleButton').prop('checked', true);
        setPractice();

        $('.code.example').hide();
        $('#canvas2GPU').hide();
        $('.test_case').hide();
        $('#image2GPU').hide();
        $('#data_div').hide();
        if (webCS == null)
        {
            webCS = await WebCS.create({ canvas: $('#canvas2GPU')[0] });
            //webCS = new WebCS();
        }
 
        let ele = myfilter;
        editorgpu.setValue(gpu_kernels[ele].toString());
        editorjs.setValue((('do_' + ele in do_cs) ? do_cs['do_' + ele] : do_cs['do_general']).toString());
    }

    $('#loadexamplemenu').menu({ select: loadExample });
});
