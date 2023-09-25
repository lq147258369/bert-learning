package com.pdd.service.omaha.integration.service.impl;

import com.pdd.service.omaha.integration.service.InferenceService;
import com.pdd.service.omaha.integration.service.response.InferenceResponse;
import com.pdd.service.omaha.integration.service.response.InferenceResult;
import com.pdd.service.omaha.integration.service.response.JarvisInferenceResponse;
import com.pinduoduo.arch.queqiao.consumer.http.HttpPost;
import com.pinduoduo.arch.queqiao.consumer.http.HttpTemplate;
import com.pinduoduo.arch.queqiao.consumer.http.serializer.TypeRef;
import com.pinpinxiaozhan.service.base.utils.JsonUtils;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections.CollectionUtils;
import org.springframework.stereotype.Service;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
* @description: 在线推理服务实现
* @author: wenjueming
* @date: 2022/11/11 11:43
* @version: 1.0
*/
@Service
@Slf4j
public class InferenceServiceImpl implements InferenceService {

    private static final String PREDICT_URL = "/service/predict/multi/batch";

    private static final String PROVIDER_NAME = "captain-marvel-api";

    private static final String OMS_FEEDBACK_PREDICT_HOST = "captain-marvel-api-oms-feedback";

    private static final String OMS_FEEDBACK_PREDICT_URL = "/service/oms/feedback/predict";

    private static final String OMS_FEEDBACK_PREDICT_JARVIS_HOST = "captain-marvel-api-jarvis-oms-feedback";

    private static final String OMS_FEEDBACK_MODEL_LABEL = "feedback_content_clf";

    private static final String OMS_FEEDBACK_PREDICT_COMMON_URL = "/algorithm/model/predict";

    private final HttpTemplate httpClient = HttpTemplate.builder()
            .providerName(PROVIDER_NAME)
            .connRequestTimeout(1000)
            .socketTimeout(1000)
            .build();

    @Override
    public InferenceResponse doCatMapInference(List<String> pathList) {
        HttpPost post = httpClient.post(PROVIDER_NAME);
        post.setPath(PREDICT_URL);
        post.setBody(pathList);
        try {
            long starTime = System.currentTimeMillis();
            List<Double> response = post.execute(new TypeRef<List<Double>>() {});
            log.info("InferenceServiceImpl inference response:{}", JsonUtils.toJson(response));
            Double maxScore = Collections.max(response);
            int maxIndex = response.indexOf(maxScore);
            String result = pathList.get(maxIndex);
            log.info("InferenceServiceImpl inference max result:{}, maxScore:{}, cost time = {}", result, maxScore, System.currentTimeMillis() - starTime);
            return new InferenceResponse(pathList.get(maxIndex), maxScore);
        }catch (Exception e){
            log.error("InferenceServiceImpl inference meet error", e);
        }

        return InferenceResponse.getEmptyInferenceResponse();
    }

    @Override
    public InferenceResult<Double> doOmsFeedbackClassifyPredict(List<String> text) {
        HttpPost post = httpClient.post(PROVIDER_NAME, OMS_FEEDBACK_PREDICT_HOST)
                .setPath(OMS_FEEDBACK_PREDICT_URL)
                .setBody(text);
        try {
            List<Double> inferenceResult = post.execute(new TypeRef<List<Double>>() {});
            log.info("InferenceServiceImpl doOmsFeedbackClassifyPredict request:{}, response:{}",
                    JsonUtils.toJson(text), JsonUtils.toJson(inferenceResult));
            if(CollectionUtils.isNotEmpty(inferenceResult)){
                return InferenceResult.success(inferenceResult);
            }
        }catch (Exception e){
            log.error("InferenceServiceImpl doOmsFeedbackClassifyPredict meet error, request:{}", JsonUtils.toJson(text), e);
        }
        return InferenceResult.fail();
    }

    @Override
    public InferenceResult<Double> doOmsFeedbackClassifyPredictV2(List<String> text) {
        HttpPost post = httpClient.post(PROVIDER_NAME, OMS_FEEDBACK_PREDICT_JARVIS_HOST)
                .setPath(OMS_FEEDBACK_PREDICT_COMMON_URL);
        Map<String, Object> param = new HashMap<>(2);
        param.put("data", text);
        param.put("model", OMS_FEEDBACK_MODEL_LABEL);

        try {
            post.setBody(param);
            JarvisInferenceResponse response = post.execute(JarvisInferenceResponse.class);
            if(Objects.nonNull(response) && 0 == response.getCode() && CollectionUtils.isNotEmpty(response.getData())){
                return InferenceResult.success(response.getData());
            }
        }catch (Exception e){
            log.error("InferenceServiceImpl doOmsFeedbackClassifyPredictV2 meet error, request:{}", JsonUtils.toJson(param), e);
        }

        return InferenceResult.fail();
    }
}
