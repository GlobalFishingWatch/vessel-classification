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

import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableReference;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;

import com.google.api.client.auth.oauth2.Credential;
import com.google.api.client.googleapis.auth.oauth2.GoogleCredential;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.http.FileContent;
import com.google.api.client.http.GenericUrl;
import com.google.api.client.http.HttpContent;
// import com.google.api.client.http.json.JsonHttpContent;
import com.google.api.client.http.ByteArrayContent;
import com.google.api.client.http.HttpRequest;
import com.google.api.client.http.HttpResponse;
import com.google.api.client.http.HttpResponseException;
import com.google.api.client.http.HttpRequestFactory;
import com.google.api.client.http.HttpTransport;
import com.google.api.client.http.UriTemplate;
import com.google.api.client.json.JsonFactory;
import com.google.api.client.json.jackson2.JacksonFactory;
import com.google.api.services.discovery.Discovery;
import com.google.api.services.discovery.Discovery.Builder;
import com.google.api.services.discovery.Discovery.Apis.GetRest;
import com.google.api.services.discovery.model.RestDescription;
import com.google.api.services.discovery.model.RestMethod;
import com.google.api.services.discovery.model.JsonSchema;

import com.google.api.services.ml.v1beta1.CloudMachineLearningScopes;

import com.google.cloud.dataflow.examples.common.DataflowExampleOptions;
import com.google.cloud.dataflow.examples.common.DataflowExampleUtils;
import com.google.cloud.dataflow.examples.common.ExampleBigQueryTableOptions;
import com.google.cloud.dataflow.examples.common.ExamplePubsubTopicOptions;
import com.google.cloud.dataflow.sdk.coders.AvroCoder;
import com.google.cloud.dataflow.sdk.coders.DefaultCoder;
import com.google.cloud.dataflow.sdk.Pipeline;
import com.google.cloud.dataflow.sdk.PipelineResult;
import com.google.cloud.dataflow.sdk.io.BigQueryIO;
import com.google.cloud.dataflow.sdk.io.PubsubIO;
import com.google.cloud.dataflow.sdk.options.Default;
import com.google.cloud.dataflow.sdk.options.Description;
import com.google.cloud.dataflow.sdk.options.PipelineOptionsFactory;
import com.google.cloud.dataflow.sdk.options.Validation;
import com.google.cloud.dataflow.sdk.runners.DataflowPipelineRunner;
import com.google.cloud.dataflow.sdk.transforms.Filter;
import com.google.cloud.dataflow.sdk.transforms.GroupByKey;
import com.google.cloud.dataflow.sdk.transforms.PTransform;
import com.google.cloud.dataflow.sdk.transforms.ParDo;
import com.google.cloud.dataflow.sdk.transforms.windowing.AfterProcessingTime;
import com.google.cloud.dataflow.sdk.transforms.windowing.AfterWatermark;
// import com.google.cloud.dataflow.sdk.transforms.windowing.FixedWindows;
import com.google.cloud.dataflow.sdk.transforms.windowing.GlobalWindows;
import com.google.cloud.dataflow.sdk.transforms.windowing.IntervalWindow;
import com.google.cloud.dataflow.sdk.transforms.windowing.Repeatedly;
import com.google.cloud.dataflow.sdk.transforms.windowing.SlidingWindows;
import com.google.cloud.dataflow.sdk.transforms.windowing.Window;
import com.google.cloud.dataflow.sdk.transforms.MapElements;

import com.google.cloud.dataflow.sdk.transforms.DoFn;
import com.google.cloud.dataflow.sdk.transforms.DoFn.RequiresWindowAccess;
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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TimeZone;

import org.apache.avro.reflect.Nullable;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectMapper.DefaultTyping;
import com.fasterxml.jackson.databind.SerializationFeature;

import static java.util.stream.Collectors.toList;
import com.google.common.geometry.S2CellId;
import com.google.common.geometry.S2LatLng;
import java.io.StringWriter;
import java.io.PrintWriter;


/**
 * ...
 *
 * <p>To execute this pipeline using the Dataflow service, specify the pipeline configuration
 * like this:
 * <pre>{@code
 *   --project=YOUR_PROJECT_ID
 *   --stagingLocation=gs://YOUR_STAGING_DIRECTORY
 *   --runner=BlockingDataflowPipelineRunner
 *   --bigQueryDataset=YOUR-DATASET --bigQueryTable=YOUR-NEW-TABLE-NAME
  *  --pubsubTopic=projects/YOUR-PROJECT/topics/YOUR-TOPIC
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


  // ====various incredibly verbose pojo definitions======
  // Not least b/c an object is apparently needed every time we want to generate/read json.


  @DefaultCoder(AvroCoder.class)
  @JsonIgnoreProperties(ignoreUnknown = true)
  static class PredictionRequestInstance {
    @Nullable Integer mmsis;
    @Nullable List<Long> timestamps;
    @Nullable List<List<List<Double>>> features; // sigh

    public PredictionRequestInstance() {}

    public PredictionRequestInstance(String mmsi, List<List<List<Double>>> features,
      List<Long> timestamps) throws NumberFormatException {
      this.mmsis = Integer.parseInt(mmsi);
      this.features = features;
      this.timestamps = timestamps;
    }

    public List<Long> getTimestamps() {
      return this.timestamps;
    }
    public List<List<List<Double>>> getFeatures() {
      return features;
    }
    public Integer getMmsis() {
      return this.mmsis;
    }

  }

  @DefaultCoder(AvroCoder.class)
  @JsonIgnoreProperties(ignoreUnknown = true)
  static class PredictionRequest {
    @Nullable List<PredictionRequestInstance> instances;

    public PredictionRequest() {}

    public PredictionRequest(List<PredictionRequestInstance> instances) {
      this.instances = instances;
    }

    public List<PredictionRequestInstance> getInstances() {
      return this.instances;
    }
  }

  @DefaultCoder(AvroCoder.class)
  @JsonIgnoreProperties(ignoreUnknown = true)
  static class PredictionResults {
    @Nullable List<Prediction> predictions;

    public PredictionResults() {}

    public PredictionResults(List<Prediction> predictions) {
      this.predictions = predictions;
    }

    public List<Prediction> getPredictions() {
      return predictions;
    }
  }


  @DefaultCoder(AvroCoder.class)
  @JsonIgnoreProperties(ignoreUnknown = true)
  static class Prediction {
    @Nullable Integer mmsis;
    @Nullable List<Long> timestamps;
    @JsonProperty("fishing_scores")
    @Nullable List<Double> fishingScores;

    public Prediction() {}

    public Prediction(Integer mmsis, List<Double> fishingScores,
      List<Long> timestamps) {
      this.mmsis = mmsis;
      this.fishingScores = fishingScores;
      this.timestamps = timestamps;
    }

    public List<Long> getTimestamps() {
      return this.timestamps;
    }
    public List<Double> getFishingScores() {
      return fishingScores;
    }
    public Integer getMmsis() {
      return this.mmsis;
    }
  }

  /**
   * Class to hold info about ship movement sequence
   */
  @DefaultCoder(AvroCoder.class)
  @JsonIgnoreProperties(ignoreUnknown = true)
  static class ShipInfo {
    @Nullable String mmsi;
    @Nullable List<List<Double>> feature;
    // @Nullable Long s2CellId; // of first elt in seq
    @Nullable long firstTimestamp;
    @Nullable String firstTimestampStr;
    @Nullable long windowTimestamp;
    @Nullable String windowTimestampStr;
    @Nullable List<Long> timestampList;
    @Nullable List<List<String>> timestampsS2Ids;
    @Nullable Map<String, String> tsS2Map;

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
    public void setFeature(List<List<Double>> f) {
      this.feature = f;
    }
    public long getFirstTimestamp() {
      return this.firstTimestamp;
    }
    public void setFirstTimestamp(long ts) {
      this.firstTimestamp = ts;
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
    public void setFirstTimestampStr(String tsStr) {
      this.firstTimestampStr = tsStr;
    }
    public void setTimestampList(List<Long> tslist) {
      this.timestampList = tslist;
    }
    public List<Long> getTimestampList() {
      return this.timestampList;
    }
    public void setTimestampsS2Ids(List<List<String>> s) {
      this.timestampsS2Ids = s;
    }
    public void setTsS2Map(Map<String, String> m) {
      this.tsS2Map = m;
    }
    public Map<String, String> getTsS2Map() {
      return this.tsS2Map;
    }
    public List<List<String>> getTimestampsS2Ids() {
      return this.timestampsS2Ids;
    }

    public String toString() {
      return "mmsi: " + this.mmsi + ", ts/s2cell id list: " + this.timestampsS2Ids +
        ", tss2map: " + this.tsS2Map +
        ", first timestamp: " + this.firstTimestamp + "/" + this.firstTimestampStr +
        ", feature seq: " + this.feature + ",\ntimestampList: " + this.timestampList;
    }

  }


  // =============================

  /**
   * Options supported by {@link FishingActivity}.
   */
  interface Options extends DataflowExampleOptions, ExampleBigQueryTableOptions {

    @Description("Pub/Sub topic to read from")
    @Validation.Required
    String getPubsubTopic();
    void setPubsubTopic(String value);

    @Description("Numeric value of window duration, in hours")
    @Default.Integer(97)
    Integer getWindowDuration();
    void setWindowDuration(Integer value);

    @Description("Numeric value of window slide, in hours")
    @Default.Integer(3)
    Integer getWindowSlide();
    void setWindowSlide(Integer value);

    @Description("Numeric value of allowed data lateness, in days")
    @Default.Integer(60)
    Integer getAllowedLateness();
    void setAllowedLateness(Integer value);

  }


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

  // probably not large enough to need a CombinePerKey approach...
  static class GatherMMSIs
      extends DoFn<KV<String, Iterable<ShipInfo>>, KV<String, KV<Integer, ShipInfo>>> {

    private static final Logger LOG = LoggerFactory.getLogger(GatherMMSIs.class);

    @Override
    public void processElement(ProcessContext c) {
      int threshold = 450; // number of features in the mmsi grouped list
      // int threshold = 50; // tempcoll - reduce for testing...
      String mmsi = c.element().getKey();
      String windowTimestamp = c.timestamp().toString();
      try {
        Iterable<ShipInfo> sl = c.element().getValue();
        // temp...
        if (sl == null) {
          LOG.warn("in GatherMMSIs with null ship info list");
          return;
        }
        List<ShipInfo> sInfoList = Lists.newArrayList(c.element().getValue());
        // do an aggregate size check first off
        int totalSize = 0;
        for (ShipInfo s : sInfoList) {
          totalSize+= s.getFeature().size();
        }
        if (totalSize < threshold) {
          // temp log tracking of what's filtered out
          int emit = (int) (Math.random() * 5000);
          if (emit == 0) {
            LOG.info("in GatherMMSIs, rejecting aggregate of size " + totalSize);
          }
          return;
        }
        // otherwise, go ahead and construct aggregate..
        List<List<Double>> allFList = new ArrayList<List<Double>>();
        Map<String, String> tsS2Map = new HashMap<String, String>();
        // temp
        try {
          for (ShipInfo s : sInfoList) {
            List<List<Double>> feature = s.getFeature();
            allFList.addAll(feature);
            for (List<String> pair: s.getTimestampsS2Ids()) {
              tsS2Map.put(pair.get(0), pair.get(1));  // sigh
            }
          }
        } catch (Exception e) {
          LOG.warn("issue creating tsmap.");
          // ugh ugh ughhh
          StringWriter sw = new StringWriter();
          e.printStackTrace(new PrintWriter(sw));
          String exceptionAsString = sw.toString();
          LOG.warn("Error: " + exceptionAsString);
        }
        if (allFList.size() >= threshold) { // just filter on size here...
          // sort the aggregate features list by timestamp
          allFList.sort((e1, e2) -> Long.compare(Math.round(e1.get(0)), Math.round(e2.get(0))));
          List<Long> tslist = allFList.stream()  // create timestamp list..
                         .map(elt -> Math.round(elt.get(0)))
                         .collect(toList());

          ShipInfo aggShipInfo = new ShipInfo(mmsi, allFList);
          long fts = Math.round(allFList.get(0).get(0)) * 1000;
          aggShipInfo.setFirstTimestamp(fts);
          aggShipInfo.setFirstTimestampStr(new Instant(fts).toString());
          aggShipInfo.setTimestampList(tslist);
          aggShipInfo.setTsS2Map(tsS2Map);
          // LOG.info("aggregate sInfo: " + aggShipInfo);
          c.output(KV.of(mmsi, KV.of(allFList.size(), aggShipInfo)));
        }
      }  catch (Exception e) {
        // ugh ugh ughhh
        StringWriter sw = new StringWriter();
        e.printStackTrace(new PrintWriter(sw));
        String exceptionAsString = sw.toString();
        LOG.warn("Error: " + exceptionAsString);
      }
    }
  }

  static class WriteS2Groups
      extends DoFn<KV<String, Iterable<KV<String, KV<Long, Double>>>>,
      TableRow> implements  RequiresWindowAccess {

    private static final Logger LOG = LoggerFactory.getLogger(WriteS2Groups.class);

    @Override
    public void processElement(ProcessContext c) {
      int minSize = 10;
      String s2CellId = c.element().getKey();
      // Iterable<KV<String, KV<Long, Double>>> infoList = c.element().getValue();
      List<KV<String, KV<Long, Double>>> infoList = Lists.newArrayList(c.element().getValue());
      if (infoList.size() < minSize) {
        return;
      }
      Set<String> mmsis = new HashSet<String>();
      int count = 0;
      Long minTs = null; Long maxTs = null;
      for (KV<String, KV<Long, Double>> i : infoList) {
        Long ts = i.getValue().getKey();
        count++;
        mmsis.add(i.getKey());
        if (maxTs == null) {
          maxTs = ts;
        } else if (ts > maxTs) {
          maxTs = ts;
        }
        if (minTs == null) {
          minTs = ts;
        } else if (ts < minTs) {
          minTs = ts;
        }
      }
      if (count > minSize) {
        S2LatLng latlon = S2CellId.fromToken(s2CellId).toLatLng();
        Double lat = latlon.latDegrees();
        Double lon = latlon.lngDegrees();
        TableRow row = new TableRow()
            .set("s2_cell_id", s2CellId)
            // .set("mmsis", mmsis.toString())
            .set("mmsis", mmsis)  // mode is repeated -- so does this work?
            .set("min_time", new Instant(minTs * 1000).toString())
            .set("max_time", new Instant(maxTs * 1000).toString())
            .set("window", c.window().toString())
            .set("lat", lat).set("lon", lon)
            .set("count", count)
            .set("processing_time", Instant.now().toString());

        String val = "mmsis: " + mmsis + ", count: " + count +
          ", wts: " + c.window().toString() +
          ", min ts: " + new Instant(minTs * 1000).toString() +
          ", max ts: " + new Instant(maxTs * 1000).toString();
        LOG.info("s2cellId " + s2CellId + ", count: " + count + ", info: " + val);
        c.output(row);
      }
    }
  }

  static class CallMLAPI
      extends DoFn<KV<String, KV<Integer, ShipInfo>>, KV<String, KV<String, KV<Long, Double>>>> {

    private static final Logger LOG = LoggerFactory.getLogger(CallMLAPI.class);

    public PredictionResults makeMLCall(String mmsi,
      List<List<Double>> trimmedFeatures, List<Long> timestamps) {

      List<List<List<Double>>> featuresList = new ArrayList<List<List<Double>>>();
      featuresList.add(trimmedFeatures);
      PredictionRequestInstance pri = new PredictionRequestInstance(mmsi, featuresList, timestamps);
      List<PredictionRequestInstance> priList = new ArrayList<PredictionRequestInstance>();
      priList.add(pri);
      PredictionRequest pr = new PredictionRequest(priList);
      ObjectMapper mapper = new ObjectMapper();
      try {
        String requestJson = mapper.writeValueAsString(pr);
        // LOG.info("Got actual json for ml request: " + requestJson);
        // Now, try making the actual prediction call...
        try {
          //...
          HttpTransport httpTransport = GoogleNetHttpTransport.newTrustedTransport();
          JsonFactory jsonFactory = JacksonFactory.getDefaultInstance();
          Discovery discovery = new Discovery.Builder(httpTransport, jsonFactory, null)
                                             .build();

          RestDescription api = discovery.apis().getRest("ml", "v1beta1").execute();
          RestMethod method = api.getResources().get("projects").getMethods().get("predict");

          JsonSchema param = new JsonSchema();
          String projectId = "aju-vtests2";
          String modelId = "gfw";
          String versionId = "v2";
          param.set("name", String.format(
              "projects/%s/models/%s/versions/%s", projectId, modelId, versionId));

          GenericUrl url = new GenericUrl(UriTemplate.expand(
              api.getBaseUrl() + method.getPath(), param, true));

          String contentType = "application/json";
          // HttpContent content = new JsonHttpContent(new JacksonFactory(), pr);
          HttpContent content = ByteArrayContent.fromString(contentType, requestJson);

          GoogleCredential credential = GoogleCredential.getApplicationDefault()
            .createScoped(CloudMachineLearningScopes.all());
          HttpRequestFactory requestFactory = httpTransport.createRequestFactory(credential);
          HttpRequest request = requestFactory.buildRequest(method.getHttpMethod(), url, content);

          int numRetries = 30; // TODO - how many?
          int retryCount = 0;
          int retryInterval = 1;
          while (retryCount < numRetries) {
            try {
              long startTime = System.currentTimeMillis();
              HttpResponse resp = request.execute();
              long elapsedTime = System.currentTimeMillis() - startTime;
              int status = resp.getStatusCode();
              String response = resp.parseAsString();
              LOG.warn("Elapsed time: " + elapsedTime + ", got status " + status + " and call response: " + response);
              PredictionResults pInfo = mapper.readValue(response, PredictionResults.class);
              return pInfo;
            // } catch (HttpResponseException|javax.net.ssl.SSLHandshakeException|java.net.SocketTimeoutException e3) {
            } catch (Exception e3) {
              StringWriter sw = new StringWriter();
              e3.printStackTrace(new PrintWriter(sw));
              String exceptionAsString = sw.toString();
              LOG.warn("Sleeping " + retryInterval + ", retry count: " + retryCount +
                ": ML Call Error: " + exceptionAsString);
              Thread.sleep(retryInterval);
              retryInterval*=2;
              retryCount += 1;
            }
          }
        } catch (Exception e2) {
          StringWriter sw = new StringWriter();
          e2.printStackTrace(new PrintWriter(sw));
          String exceptionAsString = sw.toString();
          LOG.warn("Error: " + exceptionAsString);
        }
      } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
        // ugh ugh ughhh
        StringWriter sw = new StringWriter();
        e.printStackTrace(new PrintWriter(sw));
        String exceptionAsString = sw.toString();
        LOG.warn("Error: " + exceptionAsString);
      }
      return null;
    }

    public PredictionResults getMLCallResults(String mmsi,
      List<List<Double>> trimmedFeatures, List<Long> timestamps) throws java.lang.InterruptedException {

      int retryInterval = 8000;
      int numRetries = 3;
      int count = 0;

      PredictionResults predResults = makeMLCall(mmsi, trimmedFeatures, timestamps);
      while (predResults == null && count < numRetries) {
        LOG.warn("in get getMLCallResults with null result; sleeping " + retryInterval);
        Thread.sleep(retryInterval);
        retryInterval *= 2;
        predResults = makeMLCall(mmsi, trimmedFeatures, timestamps);
        count++;
      }
      return predResults;
    }

    public void processfishingScores(ProcessContext c,
      PredictionResults pr, ShipInfo si) {

      // get s2 cell id for each score > thresh..?
      Double scoreThreshold = 0.5;
      if (pr == null) {
        LOG.warn("Error: null prediction results");
        return;
      }
      Prediction p = pr.getPredictions().get(0);
      List<Long> timestamps = p.getTimestamps();
      List<Double> scores = p.getFishingScores();
      try {
        if (timestamps.size() != scores.size()) {
          LOG.warn("timestamps and scores lists not the same size: " + timestamps.size() +
            ", " + scores.size());
          return;
        }
        Map<String, String> tsS2Map = si.getTsS2Map();
        for (int i = 0; i < timestamps.size(); i++) {
          // match ts string with map key, get s2 cell id.
          Long ts = timestamps.get(i);
          Double score = scores.get(i);
          String s2CellId = tsS2Map.get(ts.toString());
          if (s2CellId == null) {
            LOG.warn("Error: should have been able to find cell id for ts " + ts);
          }
          else {
            if (score >= scoreThreshold) {
              c.output(KV.of(s2CellId, KV.of(p.getMmsis().toString(), KV.of(ts, score))));
            }
          }
        }
      } catch (Exception e) {
        e.printStackTrace();
        LOG.warn("Error: " + e);
      }
    }

    @Override
    public void processElement(ProcessContext c) {

      int seqLength = 512;
      // int seqLength = 50; // TEMP - smaller for testing

      String mmsi = c.element().getKey();
      Integer count = c.element().getValue().getKey();
      ShipInfo si = c.element().getValue().getValue();
      List<List<Double>> features = si.getFeature();
      try {
        if (count >= seqLength) {
          // ..then we have enough data to call the ML API with our prediction query.
          // We need to generate the timestamps list and trim the timestamps from the features list.
          // TODO -- confirm the features/timestamps lists are correlated as they should be.
          List<List<Double>> trimmedFeatures = features.stream()
            .map(elt -> elt.subList(1, elt.size()))
            .collect(toList());
          // LOG.info("in CallMLAPI: mmsi " + mmsi + ", count: " + count +
          //   ", trimmed features: " + trimmedFeatures + ", timestamps: " + si.getTimestampList());
          if (trimmedFeatures.size() != si.getTimestampList().size()) {
            LOG.warn("timestamps and features lists not the same size: " + si.getTimestampList().size() +
              ", " + trimmedFeatures.size());
            return;
          }
          // trim to 512 as necessary before passing to ML prediction call
          // TODO : may want to pad if close to 512.
          trimmedFeatures = trimmedFeatures.subList(0, seqLength);
          List<Long> tsList = si.getTimestampList().subList(0, seqLength);
          // (pretend to) make the prediction request.
          PredictionResults predResults = getMLCallResults(mmsi, trimmedFeatures, tsList);
          processfishingScores(c, predResults, si);
        }
      } catch (Exception e) {
        e.printStackTrace();
        LOG.warn("Error: " + e);
      }
    }
  }


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
          // .topic("projects/earth-outreach/topics/gfwfeatures2"))
          .topic(options.getPubsubTopic()))
        .apply(ParDo.named("ParsefeatureInfo").of(new ParseShipInfoFn()));

    PCollection<KV<String, Iterable<ShipInfo>>> mmsis = fishingEvents
        .apply("ExtractMMSI",
          MapElements.via((ShipInfo sInfo) -> KV.of(sInfo.getMMSI(), sInfo))
            .withOutputType(new TypeDescriptor<KV<String, ShipInfo>>() {}))
        .apply("window1", Window
              .<KV<String, ShipInfo>>into(
                SlidingWindows.of(Duration.standardHours(options.getWindowDuration()))
                .every(Duration.standardHours(options.getWindowSlide())))
              .triggering(AfterWatermark
                           .pastEndOfWindow()
                           .withLateFirings(AfterProcessingTime
                                .pastFirstElementInPane()
                                .plusDelayOf(Duration.standardMinutes(10))))
              .accumulatingFiredPanes()
              .withAllowedLateness(Duration.standardDays(options.getAllowedLateness())))
        .apply("gbk1", GroupByKey.<String, ShipInfo>create());
    PCollection<KV<String, Iterable<KV<String, KV<Long, Double>>>>> mmsiAggregates = mmsis
      .apply("gatherMMSIs", ParDo.of(new GatherMMSIs()))
      .apply("callML", ParDo.of(new CallMLAPI()))
      .apply("gbk2", GroupByKey.<String, KV<String, KV<Long, Double>>>create());
    PCollection<TableRow> tempcoll = mmsiAggregates
      .apply("writes2groups", ParDo.of(new WriteS2Groups()));
    TableReference tableRef = getTableReference(options.getProject(),
        options.getBigQueryDataset(), options.getBigQueryTable());
    tempcoll.apply(BigQueryIO.Write.to(tableRef).withSchema(getSchema()));


    // Run the pipeline.
    PipelineResult result = pipeline.run();
    // dataflowUtils.waitToFinish(result);
  }

  /**Sets the table reference. **/
  private static TableReference getTableReference(String project, String dataset, String table){
    TableReference tableRef = new TableReference();
    tableRef.setProjectId(project);
    tableRef.setDatasetId(dataset);
    tableRef.setTableId(table);
    return tableRef;
  }

  /** Defines the BigQuery schema used for the output. */
  private static TableSchema getSchema() {
    List<TableFieldSchema> fields = new ArrayList<>();
    fields.add(new TableFieldSchema().setName("s2_cell_id").setType("STRING"));
    fields.add(new TableFieldSchema().setName("lat").setType("FLOAT"));
    fields.add(new TableFieldSchema().setName("lon").setType("FLOAT"));
    fields.add(new TableFieldSchema().setName("min_time").setType("TIMESTAMP"));
    fields.add(new TableFieldSchema().setName("max_time").setType("TIMESTAMP"));
    fields.add(new TableFieldSchema().setName("window").setType("STRING"));
    fields.add(new TableFieldSchema().setName("mmsis").setType("STRING").setMode("REPEATED"));
    fields.add(new TableFieldSchema().setName("count").setType("INTEGER"));
    fields.add(new TableFieldSchema().setName("processing_time").setType("TIMESTAMP"));

    TableSchema schema = new TableSchema().setFields(fields);
    return schema;
  }

}
