package org.skytruth.dataflow;

import com.google.cloud.dataflow.sdk.io.FileBasedSink;
import com.google.cloud.dataflow.sdk.options.PipelineOptions;

import com.google.common.hash.Hashing;
import com.google.common.io.LittleEndianDataOutputStream;

import com.google.protobuf.MessageLite;
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.nio.channels.Channels;
import java.nio.channels.WritableByteChannel;


public class TFRecordSink extends FileBasedSink<MessageLite> {
  public TFRecordSink(String baseOutputFilename, String extension, String fileNameTemplate) {
    super(baseOutputFilename, extension, fileNameTemplate);
  }

  @Override
  public FileBasedSink.FileBasedWriteOperation createWriteOperation(PipelineOptions options) {
    return new TFRecordWriteOperation(this);
  }

  private static class TFRecordWriteOperation extends FileBasedWriteOperation<MessageLite> {
    private TFRecordWriteOperation(TFRecordSink sink) {
      super(sink);
    }

    @Override
    public FileBasedWriter<MessageLite> createWriter(PipelineOptions options) throws Exception {
      return new TFRecordWriter(this);
    }
  }


  private static class TFRecordWriter extends FileBasedWriter<MessageLite> {
    private OutputStream out;

    public TFRecordWriter(FileBasedWriteOperation<MessageLite> writeOperation) {
      super(writeOperation);
    }

    // TFRecord masked crc checksum.
    static private int checksum(byte[] input) {
      int crc = Hashing.crc32c().hashBytes(input).asInt();
      return ((crc >> 15) | (crc << 17)) + 0xa282ead8;
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

    @Override
    protected void prepareWrite(WritableByteChannel channel) throws Exception {
      out = Channels.newOutputStream(channel);
    }

    @Override
    public void write(MessageLite value) throws Exception { 
      byte[] protoAsBytes = value.toByteArray();
      byte[] lengthAsBytes = encodeInt(protoAsBytes.length);
      byte[] lengthHash = encodeInt(checksum(lengthAsBytes));
      byte[] contentHash = encodeInt(checksum(protoAsBytes));
      out.write(lengthAsBytes);
      out.write(lengthHash);
      out.write(protoAsBytes);
      out.write(contentHash);
    }

  }
}
