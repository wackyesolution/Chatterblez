/bug error after running core

hannels: 1
Sample width: 2 bytes
Found 55 audio chunks
Processed audio saved to: C:\ebooks\Drunkard_s_Walk_--_Pohl__Frederik_--_Pohl_eBook_Collection_Volume_1_7__2011_--_Baen_Books___Distributed_by_Simon___Schuster_--_9781101206065_--_323db6cd05f9e000785e577c2ef9ca57_--_Anna_s_Archive_chapter_DrunkardsWalksplit019.trimmed.wav

Original duration: 206.92s
New duration: 182.85s
Removed silence: 24.06s (11.6%)
Chapter written to C:\ebooks\Drunkard_s_Walk_--_Pohl__Frederik_--_Pohl_eBook_Collection_Volume_1_7__2011_--_Baen_Books___Distributed_by_Simon___Schuster_--_9781101206065_--_323db6cd05f9e000785e577c2ef9ca57_--_Anna_s_Archive_chapter_DrunkardsWalksplit019.wav
Chapter 18 read in 518.72 seconds (6 characters per second)
Running FFmpeg concat command: ffmpeg -y -nostdin -f concat -safe 0 -i C:\ebooks\Drunkard_s_Walk_--_Pohl__Frederik_\Drunkard_s_Walk_--_Pohl__Frederik_--_Pohl_eBook_Collection_Volume_1_7__2011_--_Baen_Books___Distributed_by_Simon___Schuster_--_9781101206065_--_323db6cd05f9e000785e577c2ef9ca57_--_Anna_s_Archive_wav_list.txt -c:a aac -b:a 64k -progress pipe:1 -nostats C:\ebooks\Drunkard_s_Walk_--_Pohl__Frederik_\Drunkard_s_Walk_--_Pohl__Frederik_--_Pohl_eBook_Collection_Volume_1_7__2011_--_Baen_Books___Distributed_by_Simon___Schuster_--_9781101206065_--_323db6cd05f9e000785e577c2ef9ca57_--_Anna_s_Archive.tmp.mp4
Concatenation Total Duration: 14141.28 seconds
FFmpeg CONCAT Initial STDERR: ffmpeg version N-119453-g37064b2d16-20250509 Copyright (c) 2000-2025 the FFmpeg developers
FFmpeg CONCAT Initial STDERR: built with gcc 14.2.0 (crosstool-NG 1.27.0.18_7458341)
FFmpeg CONCAT Initial STDERR: configuration: --prefix=/ffbuild/prefix --pkg-config-flags=--static --pkg-config=pkg-config --cross-prefix=x86_64-w64-mingw32- --arch=x86_64 --target-os=mingw32 --enable-gpl --enable-version3 --disable-debug --enable-shared --disable-static --disable-w32threads --enable-pthreads --enable-iconv --enable-zlib --enable-libfribidi --enable-gmp --enable-libxml2 --enable-lzma --enable-fontconfig --enable-libharfbuzz --enable-libfreetype --enable-libvorbis --enable-opencl --disable-libpulse --enable-libvmaf --disable-libxcb --disable-xlib --enable-amf --enable-libaom --enable-libaribb24 --enable-avisynth --enable-chromaprint --enable-libdav1d --enable-libdavs2 --enable-libdvdread --enable-libdvdnav --disable-libfdk-aac --enable-ffnvcodec --enable-cuda-llvm --enable-frei0r --enable-libgme --enable-libkvazaar --enable-libaribcaption --enable-libass --enable-libbluray --enable-libjxl --enable-libmp3lame --enable-libopus --enable-librist --enable-libssh --enable-libtheora --enable-libvpx --enable-libwebp --enable-libzmq --enable-lv2 --enable-libvpl --enable-openal --enable-liboapv --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenh264 --enable-libopenjpeg --enable-libopenmpt --enable-librav1e --enable-librubberband --enable-schannel --enable-sdl2 --enable-libsnappy --enable-libsoxr --enable-libsrt --enable-libsvtav1 --enable-libtwolame --enable-libuavs3d --disable-libdrm --enable-vaapi --enable-libvidstab --enable-vulkan --enable-libshaderc --enable-libplacebo --enable-libvvenc --enable-libx264 --enable-libx265 --enable-libxavs2 --enable-libxvid --enable-libzimg --enable-libzvbi --extra-cflags=-DLIBTWOLAME_STATIC --extra-cxxflags= --extra-libs=-lgomp --extra-ldflags=-pthread --extra-ldexeflags= --cc=x86_64-w64-mingw32-gcc --cxx=x86_64-w64-mingw32-g++ --ar=x86_64-w64-mingw32-gcc-ar --ranlib=x86_64-w64-mingw32-gcc-ranlib --nm=x86_64-w64-mingw32-gcc-nm --extra-version=20250509
FFmpeg CONCAT Initial STDERR: libavutil      60.  2.100 / 60.  2.100
FFmpeg CONCAT Initial STDERR: libavcodec     62.  3.101 / 62.  3.101
FFmpeg CONCAT Initial STDERR: libavformat    62.  0.102 / 62.  0.102
FFmpeg CONCAT Initial STDERR: libavdevice    62.  0.100 / 62.  0.100
FFmpeg CONCAT Initial STDERR: libavfilter    11.  0.100 / 11.  0.100
FFmpeg CONCAT Initial STDERR: libswscale      9.  0.100 /  9.  0.100
FFmpeg CONCAT Initial STDERR: libswresample   6.  0.100 /  6.  0.100
FFmpeg CONCAT Initial STDERR: Input #0, concat, from 'C:\ebooks\Drunkard_s_Walk_--_Pohl__Frederik_\Drunkard_s_Walk_--_Pohl__Frederik_--_Pohl_eBook_Collection_Volume_1_7__2011_--_Baen_Books___Distributed_by_Simon___Schuster_--_9781101206065_--_323db6cd05f9e000785e577c2ef9ca57_--_Anna_s_Archive_wav_list.txt':
FFmpeg CONCAT Initial STDERR: Duration: N/A, start: 0.000000, bitrate: 128 kb/s
FFmpeg CONCAT Initial STDERR: Stream #0:0: Audio: aac (LC) ([255][0][0][0] / 0x00FF), 24000 Hz, mono, fltp, 128 kb/s
FFmpeg CONCAT Initial STDERR: Stream mapping:
FFmpeg CONCAT Initial STDERR: Stream #0:0 -> #0:0 (aac (native) -> aac (native))
FFmpeg CONCAT Initial STDERR: Output #0, mp4, to 'C:\ebooks\Drunkard_s_Walk_--_Pohl__Frederik_\Drunkard_s_Walk_--_Pohl__Frederik_--_Pohl_eBook_Collection_Volume_1_7__2011_--_Baen_Books___Distributed_by_Simon___Schuster_--_9781101206065_--_323db6cd05f9e000785e577c2ef9ca57_--_Anna_s_Archive.tmp.mp4':
FFmpeg CONCAT Initial STDERR: Metadata:
FFmpeg CONCAT Initial STDERR: encoder         : Lavf62.0.102
FFmpeg CONCAT Initial STDERR: Stream #0:0: Audio: aac (LC) (mp4a / 0x6134706D), 24000 Hz, mono, fltp, 64 kb/s
FFmpeg CONCAT Initial STDERR: Metadata:
FFmpeg CONCAT Initial STDERR: encoder         : Lavc62.3.101 aac
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Input buffer exhausted before END element found
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Decoding error: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 3.14 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Reserved bit set.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Prediction is not allowed in AAC-LC.
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] More than one AAC RDB per ADTS frame is not implemented. Update your FFmpeg version to the newest one from Git. If the problem still occurs, it means that your file has a feature which has not been implemented.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 1.12 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] SBR was found before the first channel element.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Sample rate index in program config element does not match the sample rate index configured by the container.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 1.6 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 1.12 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Sample rate index in program config element does not match the sample rate index configured by the container.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 3.4 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Assuming an incorrectly encoded 7.1 channel layout instead of a spec-compliant 7.1(wide) layout, use -strict 1 to decode according to the specification instead.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 1.10 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Reserved bit set.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Prediction is not allowed in AAC-LC.
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 2.14 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] SBR was found before the first channel element.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Sample rate index in program config element does not match the sample rate index configured by the container.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Number of bands (10) exceeds limit (9).
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 3.13 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 2.8 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Reserved bit set.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] invalid band type
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Number of scalefactor bands in group (50) exceeds limit (47).
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Sample rate index in program config element does not match the sample rate index configured by the container.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Number of bands (27) exceeds limit (26).
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Reserved bit set.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Number of bands (30) exceeds limit (23).
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Sample rate index in program config element does not match the sample rate index configured by the container.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Reserved bit set.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Number of bands (18) exceeds limit (13).
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Reserved bit set.
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] Prediction is not allowed in AAC-LC.
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 3.11 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input
FFmpeg CONCAT Initial STDERR: [aac @ 000002cc59a22300] channel element 3.1 is not allocated
FFmpeg CONCAT Initial STDERR: [aist#0:0/aac @ 000002cc57d8fe40] [dec:aac @ 000002cc59a698c0] Error submitting packet to decoder: Invalid data found when processing input