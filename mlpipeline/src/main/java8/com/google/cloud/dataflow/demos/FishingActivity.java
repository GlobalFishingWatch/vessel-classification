/*
 * Copyright (C) 2017 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package com.google.cloud.dataflow.demos;

import com.google.cloud.dataflow.examples.common.DataflowExampleOptions;
import com.google.cloud.dataflow.examples.common.DataflowExampleUtils;
import com.google.cloud.dataflow.sdk.Pipeline;
import com.google.cloud.dataflow.sdk.PipelineResult;
import com.google.cloud.dataflow.sdk.io.PubsubIO;
import com.google.cloud.dataflow.sdk.options.Default;
import com.google.cloud.dataflow.sdk.options.Description;
import com.google.cloud.dataflow.sdk.options.PipelineOptionsFactory;
import com.google.cloud.dataflow.sdk.options.Validation;
import com.google.cloud.dataflow.sdk.runners.DataflowPipelineRunner;
import com.google.cloud.dataflow.sdk.transforms.GroupByKey;
import com.google.cloud.dataflow.sdk.transforms.PTransform;
import com.google.cloud.dataflow.sdk.transforms.ParDo;
import com.google.cloud.dataflow.sdk.transforms.windowing.AfterProcessingTime;
import com.google.cloud.dataflow.sdk.transforms.windowing.AfterWatermark;
import com.google.cloud.dataflow.sdk.transforms.windowing.FixedWindows;
import com.google.cloud.dataflow.sdk.transforms.windowing.GlobalWindows;
import com.google.cloud.dataflow.sdk.transforms.windowing.IntervalWindow;
import com.google.cloud.dataflow.sdk.transforms.windowing.Repeatedly;
import com.google.cloud.dataflow.sdk.transforms.windowing.Window;
import com.google.cloud.dataflow.sdk.transforms.MapElements;
import com.google.cloud.dataflow.sdk.values.KV;
import com.google.cloud.dataflow.sdk.values.PCollection;
import com.google.cloud.dataflow.sdk.values.TypeDescriptor;
import com.google.common.annotations.VisibleForTesting;
import org.joda.time.DateTimeZone;
import org.joda.time.Duration;
import org.joda.time.Instant;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TimeZone;

import org.apache.avro.reflect.Nullable;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectMapper.DefaultTyping;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.google.cloud.dataflow.sdk.coders.AvroCoder;
import com.google.cloud.dataflow.sdk.coders.DefaultCoder;
import com.google.cloud.dataflow.sdk.transforms.DoFn;



/**
 * ...
 *
 * <p>To execute this pipeline using the Dataflow service, specify the pipeline configuration
 * like this:
 * <pre>{@code
 *   --project=YOUR_PROJECT_ID
 *   --stagingLocation=gs://YOUR_STAGING_DIRECTORY
 *   --runner=BlockingDataflowPipelineRunner
  *   --topic=projects/YOUR-PROJECT/topics/YOUR-TOPIC
 * }
 * </pre>
 * where the BigQuery dataset you specify must already exist.
 * The PubSub topic you specify should be the same topic to which the Injector is publishing.
 */
public class FishingActivity {

  private static final String PUBSUB_TIMESTAMP_LABEL_KEY = "timestamp";


  private static DateTimeFormatter fmt =
      DateTimeFormat.forPattern("yyyy-MM-dd HH:mm:ss.SSS")
          .withZone(DateTimeZone.forTimeZone(TimeZone.getTimeZone("PST")));
  static final Duration FIVE_MINUTES = Duration.standardMinutes(5);
  static final Duration TEN_MINUTES = Duration.standardMinutes(10);

  /**
   * Class to hold info about ship movement sequence
   */
  @DefaultCoder(AvroCoder.class)
  static class ShipInfo {
    @Nullable String mmsi;
    @Nullable List<List<Double>> feature;
    @Nullable Long s2CellId; // of first elt in seq
    @Nullable long firstTimestamp;
    @Nullable String firstTimestampStr;
    @Nullable long windowTimestamp;
    @Nullable String windowTimestampStr;

    public ShipInfo() {}

    public ShipInfo(String mmsi, List<List<Double>> feature) {
      this.mmsi = mmsi;
      this.feature = feature;
    }

    public String getMMSI() {
      return this.mmsi;
    }
    public List<List<Double>> getFeature() {
      return this.feature;
    }
    public long getFirstTimestamp() {
      return this.firstTimestamp;
    }
    public long getWindowTimestamp() {
      return this.windowTimestamp;
    }
    public String getWindowTimestampStr() {
      return this.windowTimestampStr;
    }
    public String getFirstTimestampStr() {
      return this.firstTimestampStr;
    }
    public void setS2CellId(Long s) {
      this.s2CellId = s;
    }
    public Long getS2CellId() {
      return this.s2CellId;
    }

    public String toString() {
      return "mmsi: " + this.mmsi + ", s2cell id: " + this.s2CellId +
        ", first timestamp" + this.firstTimestamp +
        ", feature seq: " + this.feature;
    }

  }


  /**
   * Options supported by {@link LeaderBoard}.
   */
  interface Options extends DataflowExampleOptions {

    @Description("Pub/Sub topic to read from")
    @Validation.Required
    String getTopic();
    void setTopic(String value);

    @Description("Numeric value of fixed window duration for team analysis, in minutes")
    @Default.Integer(60)
    Integer getTeamWindowDuration();
    void setTeamWindowDuration(Integer value);

    @Description("Numeric value of allowed data lateness, in minutes")
    @Default.Integer(120)
    Integer getAllowedLateness();
    void setAllowedLateness(Integer value);

  }

  // static class AddTimestampFn extends DoFn<ShipInfo, ShipInfo> {

  //   // AddTimestampFn() {
  //     // this.minTimestamp = new Instant(System.currentTimeMillis());
  //   // }

  //   @Override
  //   public void processElement(ProcessContext c) {
  //     // get timestamp from data element
  //     ShipInfo sInfo = c.element();
  //     long timestamp = sInfo.getFirstTimestamp();
  //     // long randMillis = (long) (Math.random() * RAND_RANGE.getMillis());
  //     // Instant randomTimestamp = minTimestamp.plus(randMillis);
  //     /**
  //      * Concept #2: Set the data element with that timestamp.
  //      */
  //     c.outputWithTimestamp(c.element(), new Instant(timestamp));
  //   }
  // }

  static class ParseShipInfoFn extends DoFn<String, ShipInfo> {

    private static final Logger LOG = LoggerFactory.getLogger(ParseShipInfoFn.class);

    @Override
    public void processElement(ProcessContext c) {
      String jsonString = c.element();
      // LOG.info("json string: " + jsonString);
      ObjectMapper mapper = new ObjectMapper();
      try {
      	ShipInfo sInfo = mapper.readValue(jsonString, ShipInfo.class);
      	// LOG.info("sInfo: " + sInfo);
        c.output(sInfo);
      } catch (java.io.IOException e) {
      	LOG.warn("couldn't parse string to json.." + e);
      }
    }
  }

  static class FishingMLFilter extends DoFn<ShipInfo, ShipInfo> {

    private static final Logger LOG = LoggerFactory.getLogger(FishingMLFilter.class);

    @Override
    public void processElement(ProcessContext c) {
      // ShipInfo sInfo = c.element();
      // Fake this for now: randomly decide if an elt passes the filter.
      // TODO -- integrate Cloud ML online prediction service API call
      int emit = (int) (Math.random() * 3);
      if (emit == 0) {
        LOG.info("emitting (randomly, for now): " + c.element());
        c.output(c.element());
      }
    }
  }

  static class GatherStatsMMSI
      extends DoFn<KV<String, Iterable<ShipInfo>>, KV<String, Integer>> {

    private static final Logger LOG = LoggerFactory.getLogger(GatherStatsMMSI.class);

    @Override
    public void processElement(ProcessContext c) {
      String mmsi = c.element().getKey();
      String windowTimestamp = c.timestamp().toString();
      // Iterable<ShipInfo> sInfoList = c.element().getValue();
      List<ShipInfo> sInfoList = Lists.newArrayList(c.element().getValue());

      List<List<Double>> allList = new ArrayList<List<Double>>();
      // int fsize = 0;
      for (ShipInfo s : sInfoList) {
        List<List<Double>> feature = s.getFeature();
        // fsize += feature.size();
        allList.addAll(feature);
        LOG.info("mmsi " + mmsi + ", s2cell id " + s.getS2CellId() + ", feature list of size : " + feature.size());
      }
      allList.sort((e1, e2) -> Long.compare(Math.round(e1.get(0)), Math.round(e2.get(0))));
      // TODO: create new shipinfo object from sorted list. How to define s2cell id?

      LOG.info("total count for: mmsi" + mmsi + " in window " + windowTimestamp + ": " + sInfoList.size());
      LOG.info("total fsize for: mmsi" + mmsi + " in window " + windowTimestamp + ": " + allList.size());
      LOG.info("sorted list: " + allList);
    }
  }

  // static class GatherStatsS2
  //     extends DoFn<KV<Long, Iterable<ShipInfo>>, KV<Long, Integer>> {

  //   private static final Logger LOG = LoggerFactory.getLogger(GatherStatsS2.class);

  //   @Override
  //   public void processElement(ProcessContext c) {
  //     Long s2CellId = c.element().getKey();
  //     List<ShipInfo> sInfoList = Lists.newArrayList(c.element().getValue());
  //     Integer count = sInfoList.size();
  //     LOG.info("for s2 cell id " + s2CellId + ", count is: " + count);
  //     c.output(KV.of(s2CellId, count));
  //   }
  // }



  public static void main(String[] args) throws Exception {

    Options options = PipelineOptionsFactory.fromArgs(args).withValidation().as(Options.class);
    // Enforce that this pipeline is always run in streaming mode.
    options.setStreaming(true);
    // For example purposes, allow the pipeline to be easily cancelled instead of running
    // continuously.
    options.setRunner(DataflowPipelineRunner.class);
    DataflowExampleUtils dataflowUtils = new DataflowExampleUtils(options);
    Pipeline pipeline = Pipeline.create(options);

    // Read game events from Pub/Sub using custom timestamps, which are extracted from the pubsub
    // data elements, and parse the data.
    PCollection<ShipInfo> fishingEvents = pipeline
        .apply(PubsubIO.Read
          .timestampLabel(PUBSUB_TIMESTAMP_LABEL_KEY)
          // .topic("projects/aju-vtests2/topics/gfwfeatures2"))
          .topic("projects/earth-outreach/topics/gfwfeatures2"))
        .apply(ParDo.named("ParsefeatureInfo").of(new ParseShipInfoFn()));
        // .apply(ParDo.of(new AddTimestampFn()));
        // .apply(ParDo.named("DetectFishingActivity").of(new FishingMLFilter()));

    PCollection<KV<String, Iterable<ShipInfo>>> mmsis = fishingEvents
        .apply("ExtractMMSI",
          MapElements.via((ShipInfo sInfo) -> KV.of(sInfo.getMMSI(), sInfo))
            .withOutputType(new TypeDescriptor<KV<String, ShipInfo>>() {}))
        .apply("window1", Window
              .<KV<String, ShipInfo>>into(
                FixedWindows.of(Duration.standardHours(49)))
              .triggering(AfterWatermark
                           .pastEndOfWindow()
                           .withLateFirings(AfterProcessingTime
                                .pastFirstElementInPane()
                                .plusDelayOf(Duration.standardMinutes(10))))
              .discardingFiredPanes()
              .withAllowedLateness(Duration.standardDays(60)))  // aju TODO: fix this
        .apply(GroupByKey.<String, ShipInfo>create());
    PCollection<KV<String, Integer>> mmsiStats = mmsis.apply(ParDo.of(new GatherStatsMMSI()));


    // PCollection<KV<Long, Iterable<ShipInfo>>> s2cells = fishingEvents
    //     .apply("ExtractS2Features",
    //       MapElements.via((ShipInfo sInfo) -> KV.of(sInfo.getS2CellId(), sInfo))
    //         .withOutputType(new TypeDescriptor<KV<Long, ShipInfo>>() {}))
    //     .apply("window1", Window
    //           .<KV<Long, ShipInfo>>into(FixedWindows.of(Duration.standardMinutes(5))))
    //     .apply(GroupByKey.<Long, ShipInfo>create());
    // PCollection<KV<Long, Integer>> s2CellsStats = s2cells.apply(ParDo.of(new GatherStatsS2()));


    // Run the pipeline and wait for the pipeline to finish; capture cancellation requests from the
    // command line.
    PipelineResult result = pipeline.run();
    dataflowUtils.waitToFinish(result);
  }


}
