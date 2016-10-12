package org.skytruth.dataflow;

import com.google.common.hash.Hashing;
import com.google.common.io.LittleEndianDataOutputStream;

import com.google.protobuf.MessageLite;
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;

public class TFRecordUtils {
  // TFRecord masked crc checksum.
  static private int checksum(byte[] input) {
    int crc = Hashing.crc32c().hashBytes(input).asInt();
    return ((crc >>> 15) | (crc << 17)) + 0xa282ead8;
  }

  static private byte[] encodeLong(long in) throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    LittleEndianDataOutputStream out = new LittleEndianDataOutputStream(baos);
    out.writeLong(in);
    return baos.toByteArray();
  }
  
  static private byte[] encodeInt(int in) throws Exception {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    LittleEndianDataOutputStream out = new LittleEndianDataOutputStream(baos);
    out.writeInt(in);
    return baos.toByteArray();
  }

  static public void write(OutputStream out, MessageLite value) throws Exception {
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