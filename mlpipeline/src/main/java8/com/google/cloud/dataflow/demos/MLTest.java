
package com.google.cloud.dataflow.demos;


import com.google.api.client.auth.oauth2.Credential;
import com.google.api.client.googleapis.auth.oauth2.GoogleCredential;
import com.google.api.client.googleapis.javanet.GoogleNetHttpTransport;
import com.google.api.client.http.FileContent;
import com.google.api.client.http.GenericUrl;
import com.google.api.client.http.HttpContent;
import com.google.api.client.http.json.JsonHttpContent;
import com.google.api.client.http.HttpRequest;
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


import java.io.File;

public class MLTest {

    public static void main(String[] args) throws Exception {
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
        File requestBodyFile = new File("/Users/amyu/devrel/vessel-classification-pipeline-fork/mlpipeline/instances2.json");
        HttpContent content = new FileContent(contentType, requestBodyFile);
        // HttpContent content = new JsonHttpContent(new JacksonFactory(), data);

        GoogleCredential credential = GoogleCredential.getApplicationDefault().createScoped(CloudMachineLearningScopes.all());
        HttpRequestFactory requestFactory = httpTransport.createRequestFactory(credential);
        HttpRequest request = requestFactory.buildRequest(method.getHttpMethod(), url, content);

        String response = request.execute().parseAsString();
        System.out.println(response);
    }
}
