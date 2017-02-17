// Copyright 2017 Google Inc. and Skytruth Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package org.skytruth.dataflow;

import com.google.common.hash.Hashing;
import com.google.common.io.LittleEndianDataOutputStream;
import com.google.protobuf.MessageLite;
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;

public class TFRecordUtils {
  // TFRecord masked crc checksum.
  private static int checksum(byte[] input) {
    int crc = Hashing.crc32c().hashBytes(input).asInt();
    return ((crc >>> 15) | (crc << 17)) + 0xa282ead8;
  }

  private static byte[] encodeLong(long in) throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    LittleEndianDataOutputStream out = new LittleEndianDataOutputStream(baos);
    out.writeLong(in);
    return baos.toByteArray();
  }

  private static byte[] encodeInt(int in) throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    LittleEndianDataOutputStream out = new LittleEndianDataOutputStream(baos);
    out.writeInt(in);
    return baos.toByteArray();
  }

  public static void write(OutputStream out, MessageLite value) throws Exception {
    byte[] protoAsBytes = value.toByteArray();
    byte[] lengthAsBytes = encodeLong(protoAsBytes.length);
    byte[] lengthHash = encodeInt(checksum(lengthAsBytes));
    byte[] contentHash = encodeInt(checksum(protoAsBytes));
    out.write(lengthAsBytes);
    out.write(lengthHash);
    out.write(protoAsBytes);
    out.write(contentHash);
  }
}
