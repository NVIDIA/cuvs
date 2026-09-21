/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs;

import java.util.Objects;

/**
 * Caller-owned tensors describing one encoded BBQ dataset representation. Keep the component
 * matrices open while a graph-only CAGRA index has this representation attached.
 */
public final class BbqQuantizer {
  public enum CodeLayout {
    PACKED_1B(0),
    TRANSPOSED_2B(1),
    TRANSPOSED_4B(2),
    PACKED_4B(3),
    PACKED_7B(4),
    PACKED_8B(5);

    public final int value;

    CodeLayout(int value) {
      this.value = value;
    }
  }

  private final CuVSMatrix codes;
  private final CuVSMatrix lowerIntervals;
  private final CuVSMatrix upperIntervals;
  private final CuVSMatrix additionalCorrections;
  private final CuVSMatrix quantizedComponentSums;
  private final CuVSMatrix centroid;
  private final CuVSMatrix dequantDelta;
  private final CuVSMatrix dequantSumDelta;
  private final CuVSMatrix rowNorm;
  private final CodeLayout layout;
  private final CagraIndexParams.CuvsDistanceType metric;
  private final float centroidNormSq;

  private BbqQuantizer(Builder builder) {
    codes = Objects.requireNonNull(builder.codes);
    lowerIntervals = Objects.requireNonNull(builder.lowerIntervals);
    upperIntervals = Objects.requireNonNull(builder.upperIntervals);
    additionalCorrections = Objects.requireNonNull(builder.additionalCorrections);
    quantizedComponentSums = Objects.requireNonNull(builder.quantizedComponentSums);
    centroid = Objects.requireNonNull(builder.centroid);
    dequantDelta = Objects.requireNonNull(builder.dequantDelta);
    dequantSumDelta = Objects.requireNonNull(builder.dequantSumDelta);
    rowNorm = Objects.requireNonNull(builder.rowNorm);
    layout = Objects.requireNonNull(builder.layout);
    metric = Objects.requireNonNull(builder.metric);
    centroidNormSq = builder.centroidNormSq;
  }

  public CuVSMatrix getCodes() {
    return codes;
  }

  public CuVSMatrix getLowerIntervals() {
    return lowerIntervals;
  }

  public CuVSMatrix getUpperIntervals() {
    return upperIntervals;
  }

  public CuVSMatrix getAdditionalCorrections() {
    return additionalCorrections;
  }

  public CuVSMatrix getQuantizedComponentSums() {
    return quantizedComponentSums;
  }

  public CuVSMatrix getCentroid() {
    return centroid;
  }

  public CuVSMatrix getDequantDelta() {
    return dequantDelta;
  }

  public CuVSMatrix getDequantSumDelta() {
    return dequantSumDelta;
  }

  public CuVSMatrix getRowNorm() {
    return rowNorm;
  }

  public CodeLayout getLayout() {
    return layout;
  }

  public CagraIndexParams.CuvsDistanceType getMetric() {
    return metric;
  }

  public float getCentroidNormSq() {
    return centroidNormSq;
  }

  public static final class Builder {
    private CuVSMatrix codes;
    private CuVSMatrix lowerIntervals;
    private CuVSMatrix upperIntervals;
    private CuVSMatrix additionalCorrections;
    private CuVSMatrix quantizedComponentSums;
    private CuVSMatrix centroid;
    private CuVSMatrix dequantDelta;
    private CuVSMatrix dequantSumDelta;
    private CuVSMatrix rowNorm;
    private CodeLayout layout;
    private CagraIndexParams.CuvsDistanceType metric;
    private float centroidNormSq;

    public Builder withCodes(CuVSMatrix value) {
      codes = value;
      return this;
    }

    public Builder withLowerIntervals(CuVSMatrix value) {
      lowerIntervals = value;
      return this;
    }

    public Builder withUpperIntervals(CuVSMatrix value) {
      upperIntervals = value;
      return this;
    }

    public Builder withAdditionalCorrections(CuVSMatrix value) {
      additionalCorrections = value;
      return this;
    }

    public Builder withQuantizedComponentSums(CuVSMatrix value) {
      quantizedComponentSums = value;
      return this;
    }

    public Builder withCentroid(CuVSMatrix value) {
      centroid = value;
      return this;
    }

    public Builder withDequantDelta(CuVSMatrix value) {
      dequantDelta = value;
      return this;
    }

    public Builder withDequantSumDelta(CuVSMatrix value) {
      dequantSumDelta = value;
      return this;
    }

    public Builder withRowNorm(CuVSMatrix value) {
      rowNorm = value;
      return this;
    }

    public Builder withLayout(CodeLayout value) {
      layout = value;
      return this;
    }

    public Builder withMetric(CagraIndexParams.CuvsDistanceType value) {
      metric = value;
      return this;
    }

    public Builder withCentroidNormSq(float value) {
      centroidNormSq = value;
      return this;
    }

    public BbqQuantizer build() {
      return new BbqQuantizer(this);
    }
  }
}
