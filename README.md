# webcs

Library for WebGL2-Compute

## Demo

http://zhenghuadai.github.io/webCS.html

## Practise

http://blog.biosuefi.com/webCS.html#practise

## Tutorials
### Enable WebGL 2.0 Compute in Chrome
    chrome://flags/#enable-webgl2-compute-context
### Use in browser
<script src='http://zhenghuadai.github.io/js/webcs.js'></script>                    

### Example of matrix multiply
```csharp
       function sgemmv1(A,B,C){
            return `
                // C[M, N] = A[M, K] * B[K, N]
                uint M = this.uniform.MNK.x, N = this.uniform.MNK.y, K = this.uniform.MNK.z;
                // Compute a single element C[thread.y, thread.x] by looping over k
                float sum = 0.0;
                for (uint k = 0u; k < K; k++)
                {
                    sum += A[thread.y * K + k] * B[k * N + thread.x];
                }
         
                // Store the result
                C[thread.y*N + thread.x] = sum;
            `;
        }
```
```javascript
        let M = 64, N = 64, K =64;
        var createArray = function ( n) { 
            var buf = new Float32Array(n);
            for(var i = 0; i < n; i++){buf[i] = Math.random();}
            return buf;
        };
        let cpuA = createArray(M*K);
        let cpuB = createArray(K*N);
        let cpuC = createArray(M*N);
        let webCS = new WebCS();
        let cs_sgemm = webCS.createShader(sgemmv1, {local_size:[8, 8, 1], groups:[M/8, N/8, 1]});
        
        cs_sgemm.setUniform('MNK', M, N, K, 0).run(cpuA, cpuB, cpuC);
        // or
        // cs_sgemm.run(cpuA, cpuB, cpuC, {'MNK':[M,N,K,0]});
        
        cs_sgemm.getData('C', cpuC);
```
### Example of processing image 
```csharp
        function glsl_texture2(src, dst){
            return `
            ivec2 pos = ivec2(thread.xy);
            // vec4 pixel = imageLoad(src, pos); or vec4 pixel = src[pos];
            vec4 pixel = src[pos.y][pos.x];       
            vec4 invert = vec4(1.0 - pixel.x, 1.0 - pixel.y, 1.0 - pixel.z, 1.0);
            dst[pos.y][pos.x] = invert;  // imageStore(dst, pos, invert); or dst[pos] = invert;
            `;
        }
```
```javascript
        var X = 512, Y = 512, Z = 1;
        let webCS = new WebCS({width:X, height:Y});
        // or let webCS = new WebCS({canvas:$("#canvas2GPU")[0]});
        let cs_texture2 = webCS.createShader(glsl_texture2, 
                                            { local_size:[8, 8, 1],  groups:[X/8, Y/8, 1],
                                              params:{src:'texture', 'dst':'texture'}
                                            });

        let texSrc = $('#image000')[0];
        cs_texture2.run(texSrc, null);

        let tex = cs_texture2.getTexture('dst');
        webCS.present(tex);
        $("#display1")[0].appendChild(webCS.canvas);
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

## License

Code is under [MIT](http://davidsonfellipe.mit-license.org) license

