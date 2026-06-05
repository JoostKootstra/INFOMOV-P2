# INFOMOV P2
Made by Joost Kootstra (4373189) and Justin Timmer (1126458).

# Build instructions
The project already comes with a pre-built solution. The solution, when opened, should run out-of-the-box when using Visual Studio 2022. It might also work out-of-the-box with VS 2026, but this has not been tested.

# Run instructions
In order to run the project, just click "Run" (or "Local Windows Debugger") at the top of the screen in Release Mode.
Unfortunately, we were unable to get the GPGPU version to run correctly on the data structured for SIMD. 
To still be able to hand in a working project, we decided to separate the SIMD and GPGPU functionality using conditional blocks. 
If you want to run the gpu implementation, line 40 should say "#if 1". 
If you want to run the SIMD implementation, line 40 should say "#if 0".

# Speed up
- Initial speed: average 20ms with high peaks at 23ms
- SIMD: average 15ms with high peaks at 17ms
- GPGPU: consistently between 1.3ms and 1.5ms