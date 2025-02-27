#include <jxl/decode.h>
#include <jxl/codestream_header.h>
#include <jxl/decode_cxx.h>
#include <jxl/resizable_parallel_runner.h>
#include <jxl/resizable_parallel_runner_cxx.h>
#include <jxl/types.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

struct ImageData {
    uint32_t width;
    uint32_t height;
    std::vector<uint8_t> pixels; // Interleaved RGB values
};

ImageData decodeJXL(const py::bytes &data) {
    std::string data_str = static_cast<std::string>(data);
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(data_str.data());
    const size_t inputSize = data_str.size();
    ImageData imageData = {};

    JxlDecoder* decoder = JxlDecoderCreate(nullptr);
    if (JXL_DEC_SUCCESS != JxlDecoderSubscribeEvents(decoder, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE)) {
        throw std::runtime_error("JxlDecoderSubscribeEvents failed");
    }

    JxlBasicInfo info;
    JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

    JxlDecoderSetInput(decoder, bytes, inputSize);
    JxlDecoderCloseInput(decoder);

    while (true) {
        JxlDecoderStatus status = JxlDecoderProcessInput(decoder);

        if (status == JXL_DEC_ERROR) {
            throw std::runtime_error("Decoder error");
        } else if (status == JXL_DEC_NEED_MORE_INPUT) {
            throw std::runtime_error("Error, already provided all input");
        } else if (status == JXL_DEC_BASIC_INFO) {
            if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(decoder, &info)) {
                throw std::runtime_error("JxlDecoderGetBasicInfo failed");
            }
            imageData.width = info.xsize;
            imageData.height = info.ysize;
        } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
            size_t buffer_size;
            if (JXL_DEC_SUCCESS != JxlDecoderImageOutBufferSize(decoder, &format, &buffer_size)) {
                throw std::runtime_error("JxlDecoderImageOutBufferSize failed");
            }
            if (buffer_size != info.xsize * info.ysize * 3) {
                throw std::runtime_error("Invalid out buffer size");
            }
            imageData.pixels.resize(info.xsize * info.ysize * 3);
            void *pixels_buffer = static_cast<void *>(imageData.pixels.data());
            size_t pixels_buffer_size = imageData.pixels.size() * 3;
            if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(decoder, &format, pixels_buffer, pixels_buffer_size)) {
                throw std::runtime_error("JxlDecoderSetImageOutBuffer failed");
            }
        } else if (status == JXL_DEC_FULL_IMAGE) {
            // Nothing to do. Do not yet return. If the image is an animation, more
            // full frames may be decoded. This example only keeps the last one.
        } else if (status == JXL_DEC_SUCCESS) {
            // All decoding successfully finished.
            // It's not required to call JxlDecoderReleaseInput(dec.get()) here since
            // the decoder will be destroyed.
            return imageData;
        } else {
            throw std::runtime_error("Unknown decoder status");
        }
    }
}

PYBIND11_MODULE(jxl_decoder, m) {
    py::class_<ImageData>(m, "ImageData")
        .def_readwrite("width", &ImageData::width)
        .def_readwrite("height", &ImageData::height)
        .def_readwrite("pixels", &ImageData::pixels);

    m.def("decodeJXL", &decodeJXL, "Decode a JXL image from a byte array");
}