# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: com/appliedrec/verid3/serialization/capture3d/depth_map.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from com.appliedrec.verid3.serialization.common import pointf_pb2 as com_dot_appliedrec_dot_verid3_dot_serialization_dot_common_dot_pointf__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n=com/appliedrec/verid3/serialization/capture3d/depth_map.proto\x12-com.appliedrec.verid3.serialization.capture3d\x1a\x37\x63om/appliedrec/verid3/serialization/common/pointf.proto\"\xf9\x02\n\x08\x44\x65pthMap\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\r\n\x05width\x18\x02 \x01(\x05\x12\x0e\n\x06height\x18\x03 \x01(\x05\x12\x15\n\rbytes_per_row\x18\x04 \x01(\x05\x12\x18\n\x10\x62its_per_element\x18\x05 \x01(\x05\x12K\n\x0fprincipal_point\x18\x06 \x01(\x0b\x32\x32.com.appliedrec.verid3.serialization.common.PointF\x12H\n\x0c\x66ocal_length\x18\x07 \x01(\x0b\x32\x32.com.appliedrec.verid3.serialization.common.PointF\x12$\n\x1clens_distortion_lookup_table\x18\x08 \x03(\x02\x12R\n\x16lens_distortion_center\x18\t \x01(\x0b\x32\x32.com.appliedrec.verid3.serialization.common.PointFB\x03\xba\x02\x00\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'com.appliedrec.verid3.serialization.capture3d.depth_map_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\272\002\000'
  _DEPTHMAP._serialized_start=170
  _DEPTHMAP._serialized_end=547
# @@protoc_insertion_point(module_scope)
