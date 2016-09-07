package org.skytruth.dataflow;

import com.google.cloud.dataflow.sdk.io.FileBasedSink;
import com.google.cloud.dataflow.sdk.options.PipelineOptions;

import com.google.protobuf.MessageLite;
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

    @Override
    protected void prepareWrite(WritableByteChannel channel) throws Exception {
      out = Channels.newOutputStream(channel);
    }

    @Override
    public void write(MessageLite value) throws Exception {
      TFRecordUtils.write(out, value);
    }
  }
}

