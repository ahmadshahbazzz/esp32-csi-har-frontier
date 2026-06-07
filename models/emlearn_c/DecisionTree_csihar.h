


    // !!! This file is generated using emlearn !!!

    #include <stdint.h>
    

static inline int32_t DecisionTree_csihar_tree_0(const int16_t *features, int32_t features_length) {
          if (features[156] < 0) {
              if (features[254] < 2) {
                  if (features[1] < -1) {
                      if (features[45] < 1) {
                          if (features[142] < -1) {
                              return 1;
                          } else {
                              if (features[229] < 3) {
                                  if (features[202] < 1) {
                                      if (features[222] < 3) {
                                          return 5;
                                      } else {
                                          return 4;
                                      }
                                  } else {
                                      return 5;
                                  }
                              } else {
                                  return 4;
                              }
                          }
                      } else {
                          if (features[42] < 0) {
                              if (features[256] < 2) {
                                  return 4;
                              } else {
                                  return 1;
                              }
                          } else {
                              return 2;
                          }
                      }
                  } else {
                      if (features[252] < 2) {
                          if (features[133] < -1) {
                              if (features[17] < 0) {
                                  return 2;
                              } else {
                                  return 5;
                              }
                          } else {
                              if (features[73] < 0) {
                                  if (features[216] < 3) {
                                      return 2;
                                  } else {
                                      if (features[114] < -2) {
                                          return 1;
                                      } else {
                                          return 4;
                                      }
                                  }
                              } else {
                                  if (features[145] < 0) {
                                      return 5;
                                  } else {
                                      return 4;
                                  }
                              }
                          }
                      } else {
                          if (features[234] < 3) {
                              return 2;
                          } else {
                              if (features[192] < 1) {
                                  return 5;
                              } else {
                                  return 6;
                              }
                          }
                      }
                  }
              } else {
                  if (features[59] < 0) {
                      if (features[36] < 0) {
                          return 1;
                      } else {
                          if (features[38] < 0) {
                              if (features[79] < 0) {
                                  return 5;
                              } else {
                                  return 0;
                              }
                          } else {
                              return 2;
                          }
                      }
                  } else {
                      if (features[22] < 0) {
                          if (features[10] < 0) {
                              return 2;
                          } else {
                              if (features[158] < 0) {
                                  if (features[107] < -2) {
                                      return 1;
                                  } else {
                                      return 4;
                                  }
                              } else {
                                  return 6;
                              }
                          }
                      } else {
                          if (features[113] < -3) {
                              if (features[133] < -1) {
                                  if (features[0] < -2) {
                                      return 2;
                                  } else {
                                      if (features[138] < -2) {
                                          return 0;
                                      } else {
                                          return 4;
                                      }
                                  }
                              } else {
                                  return 1;
                              }
                          } else {
                              if (features[46] < 0) {
                                  return 6;
                              } else {
                                  if (features[162] < 0) {
                                      return 2;
                                  } else {
                                      return 0;
                                  }
                              }
                          }
                      }
                  }
              }
          } else {
              if (features[11] < 0) {
                  if (features[70] < 0) {
                      if (features[73] < 0) {
                          return 4;
                      } else {
                          return 3;
                      }
                  } else {
                      if (features[241] < 5) {
                          if (features[258] < 2) {
                              if (features[107] < -2) {
                                  return 3;
                              } else {
                                  return 2;
                              }
                          } else {
                              return 6;
                          }
                      } else {
                          return 3;
                      }
                  }
              } else {
                  if (features[102] < 0) {
                      if (features[20] < 0) {
                          return 6;
                      } else {
                          if (features[237] < 2) {
                              return 2;
                          } else {
                              if (features[216] < 4) {
                                  return 0;
                              } else {
                                  return 1;
                              }
                          }
                      }
                  } else {
                      if (features[53] < 0) {
                          if (features[203] < 1) {
                              return 5;
                          } else {
                              return 6;
                          }
                      } else {
                          if (features[83] < 1) {
                              return 3;
                          } else {
                              if (features[57] < 0) {
                                  return 3;
                              } else {
                                  return 6;
                              }
                          }
                      }
                  }
              }
          }
        }
        

int32_t DecisionTree_csihar_predict(const int16_t *features, int32_t features_length) {

        int32_t votes[7] = {0,};
        int32_t _class = -1;

        _class = DecisionTree_csihar_tree_0(features, features_length); votes[_class] += 1;
    
        int32_t most_voted_class = -1;
        int32_t most_voted_votes = 0;
        for (int32_t i=0; i<7; i++) {

            if (votes[i] > most_voted_votes) {
                most_voted_class = i;
                most_voted_votes = votes[i];
            }
        }
        return most_voted_class;
    }
    

int DecisionTree_csihar_predict_proba(const int16_t *features, int32_t features_length, float *out, int out_length) {

        int32_t _class = -1;

        for (int i=0; i<out_length; i++) {
            out[i] = 0.0f;
        }

        _class = DecisionTree_csihar_tree_0(features, features_length); out[_class] += 1.0f;
    
        // compute mean
        for (int i=0; i<out_length; i++) {
            out[i] = out[i] / 1;
        }
        return 0;
    }
    