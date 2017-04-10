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
