# Model Performance Metrics Summary

This report summarizes the Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the number of valid samples for each evaluated model. Lower MAE and RMSE values indicate better performance.

--- Results Summary ---
| Model                           |   MAE |  RMSE | Pearson's R |   Samples |
| :------------------------------ | ----: | ----: | ----------: | --------: |
| Qwen2.5-VL-7B-Instruct          | 28.26 | 49.69 |      0.1896 |       160 |
| gemini-2.0-flash                | 11.49 | 15.65 |      0.5143 |       288 |
| gemini-2.5-flash-preview-04-17  |  6.70 | 11.86 |      0.6927 |       260 |
| gemini-2.5-pro-preview-03-25    |  5.73 |  9.67 |      0.8243 |       282 |
| qwen2.5-vl-32b-instruct         |  9.37 | 15.00 |      0.4970 |       283 |

## Trends Observed

*   **Overall Performance:** The Gemini family of models demonstrates significantly lower error rates (both MAE and RMSE) compared to `Qwen2.5-VL-7B-Instruct`.
*   **Gemini Model Comparison:** Within the Gemini models, `gemini-2.5-pro-preview-03-25` achieved the lowest MAE and RMSE, indicating the highest accuracy in this evaluation. The `gemini-2.5-flash-preview-04-17` model also performs very well, with errors slightly higher than the Pro version but considerably better than the older `gemini-2.0-flash`.
*   **Sample Sizes:** The number of valid samples processed varies between models, with the Gemini models generally having a higher count than the Qwen model in this specific analysis.