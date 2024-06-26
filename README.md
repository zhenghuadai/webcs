# webcs

Library for WebGPU-Compute

## Build and Run
- npm install
- npm run dev
```shell
   > Local:    https://localhost:3000/
   > Network:  https://192.168.2.3:3000/
```
Open the above URL in Chrome canary. 
Note: Must enable *https*, otherwise webgpu won't work

## Demo

https://zhenghuadai.github.io/webCS.html

## Practise

https://blog.biosuefi.com/webcs.html#practise

## Tutorials
### Enable WebGPU in Chrome
    chrome://flags/#enable-unsafe-webgpu
### Use in browser
<script src='http://zhenghuadai.github.io/js/webcs.js'></script>                    

### Example of matrix multiply
```csharp
        function gpu_smm_naive(A,B,C){
            return `
                // C[M, N] = A[M, K] * B[K, N]
                var M:u32 = this.uniform.MNK.x;
                var N:u32 = this.uniform.MNK.y;
                var K:u32 = this.uniform.MNK.z;
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

```
```javascript
        await (async function(kernel_name) {
            var X = 512, Y = 512, Z = 1;
            let webCS = await WebCS.create({width:X, height:Y});
            // or let webCS = await WebCS.create({canvas:$("#canvas2GPU")[0]});
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
            // groups is optional
            // local_size is mandatory
            let cs_smm_naive = webCS.createShader(gpu_smm_naive, { local_size: [8, 8, 1], groups: [M / 8, N / 8, 1] });
         
            const t0 = performance.now();
            // or await cs_smm_naive.run(cpuA, cpuB, cpuC, M / 8, N / 8, 1, { 'MNK': [M, N, K, 0] });
            // or await (cs_smm_naive.setGroups(M / 8, N / 8, 1)).run(cpuA, cpuB, cpuC, { 'MNK': [M, N, K, 0] });
            await cs_smm_naive.run(cpuA, cpuB, cpuC, { 'MNK': [M, N, K, 0] });
            const t1 = performance.now();
            let t    = t1 - t0;
            $('#time').html(t.toFixed(1).toString());
            cpuC       = await cs_smm_naive.getData('C', 'float');
        })();
```
### Example of processing image 
```csharp
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
```
```javascript
        await (async function(kernel_name) {
            var X = 512, Y = 512, Z = 1;
            let webCS = await WebCS.create({width:X, height:Y});
            // or let webCS = await WebCS.create({canvas:$("#canvas2GPU")[0]});

            // groups is optional
            // local_size is mandatory
            let tex_kernel = webCS.createShader(gpu_texture2,
            { local_size: [8, 8, 1], groups: [X / 8, Y / 8, 1], params: { src: 'texture', 'dst': 'texture' } });

            let texSrc = $('#image000')[0];
            // or await (tex_kernel.setGroups(X / 8, Y / 8, 1)).run(texSrc, null);
            // or await tex_kernel.run(texSrc, null, X / 8, Y / 8, 1);
            await tex_kernel.run(texSrc, null);
  
            let tex = tex_kernel.getTexture('dst');
            webCS.present(tex);
            $('#display1')[0].appendChild(webCS.canvas);
            $(webCS.canvas).show();
        })();
```
### kernel
A compute kernel is created by webCS.createShader(kernel_function, settings) from kernel_function; 
### input
input type for a kernel:
- js TypedArray , such as Float32Array
- WebGLBuffer
- WebGLTexture
- HTMLImageElement
- HTMLCanvasElement

### output
 - getBuffer(argument_name) to return the WebGLBuffer
   - e.g. let glBufferC = cs_sgemm.getBuffer('C');
 - getData(argument_name, typedarrary_or_num_type)  to return a typedarray
   - e.g. let result = cs_sgemm.getData('C', 'float');
   - e.g. let result = new Float32Array(64*64); cs_sgemm.getData('C', result);
### uniform
  - use this.uniform. as prefix for any uniform in GLSL shader string.
### dispatch
```
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
```
## License

Code is under [MIT](http://davidsonfellipe.mit-license.org) license

