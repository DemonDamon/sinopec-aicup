# Research Log for Sinopec AI Innovation Competition Feature Engineering


## Initial Broad Search Results

### Sinopec AI Innovation Competition winning solutions feature engineering oil and gas
*   **Title**: Sinopec Wins Technological Innovation Award at the 2nd ...
    *   **Snippet**: Sinopec has won the “Best Practice for Technological Innovation” award at the 2 nd Sino-European Corporate ESG Best Practice Conference.
    *   **URL**: http://www.sinopecgroup.com/group/en/000/000/067/67552.shtml
*   **Title**: Sinopec bags 'Technological Innovation Practice' award
    *   **Snippet**: Sinopec has been honored with the “Best Scientific and Technological Innovation Practice” award at the 2nd Sino-European Corporate ESG Best Practice Conference.
    *   **URL**: https://www.indianchemicalnews.com/general/sinopec-bags-technological-innovation-practice-award-26654
*   **Title**: Sinopec Unveils Two Achievements at the 2025 World ...
    *   **Snippet**: Sinopec's achievements include AI-assisted development of polyimide gas separation materials and an intelligent R&D platform for molecular ...
    *   **URL**: http://www.sinopecgroup.com/group/en/000/000/067/67960.shtml
*   **Title**: Sinopec Wins Technological Innovation Award at 2 nd Sino ...
    *   **Snippet**: Sinopec has harnessed technological innovation as a powerful engine in its journey toward carbon peaking and carbon neutrality.
    *   **URL**: https://via.tt.se/pressmeddelande/3991080/sinopec-wins-technological-innovation-award-at-2-nd-sino-european-corporate-esg-best-practice-conference?publisherId=259167&lang=en
*   **Title**: Title: Well Log-Based Lithology Identification and Classification
    *   **Snippet**: About the "Sinopec First Artificial Intelligence Innovation Competition". This competition focuses on the energy and petrochemical industry, covering various ...
    *   **URL**: https://www.competehub.dev/en/competitions/aicup_sinopec_02

### Kaggle oil and gas production prediction spatio-temporal features
*   **Title**: predict oil and gas productions in USA
    *   **Snippet**: Explore and run machine learning code with Kaggle Notebooks | Using data from oil and gas production rate.
    *   **URL**: https://www.kaggle.com/code/ahmedelbashir99/predict-oil-and-gas-productions-in-usa
*   **Title**: Forecasting of spatio-temporal event data using a ...
    *   **Snippet**: One notable modeling method to predict spatio-temporal data is by means of ConvLSTM models.
    *   **URL**: https://www.kaggle.com/questions-and-answers/243127
*   **Title**: Deep insight: an efficient hybrid model for oil well ...
    *   **Snippet**: Deep insight: an efficient hybrid model for oil well production forecasting using spatio-temporal convolutional networks and Kolmogorov–Arnold ...
    *   **URL**: https://www.nature.com/articles/s41598-025-91412-2

### Oil and gas water drive mechanism modeling feature engineering
*   **Title**: The Defining Series: Reservoir Drive Mechanisms
    *   **Snippet**: The energy for a waterdrive system comes from a connected aquifer. As hydrocarbons are extracted, the aquifer expands, and water migrates to replace the moved ...
    *   **URL**: https://www.slb.com/resource-library/oilfield-review/defining-series/defining-reservoir-drive-mechanisms
*   **Title**: Drive mechanisms and recovery
    *   **Snippet**: Drive mechanisms are determined by the analysis of historical production data, primarily reservoir pressure data and fluid production ratios.
    *   **URL**: http://wiki.aapg.org/Drive_mechanisms_and_recovery

### Time series feature engineering oil production data daily monthly
*   **Title**: Practical Guide for Feature Engineering of Time Series Data
    *   **Snippet**: In this tutorial, we will walk through examples of these feature types for a well-known time series dataset and discuss using pandas and SQL to manually create ...
    *   **URL**: https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/
*   **Title**: A deep learning-based approach for predicting oil production
    *   **Snippet**: It has been proven that the model can predict oil production accurately and outperform other models with a correlation coefficient reaching 0.99874.
    *   **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0360544223030827

### Perforation event feature engineering oil wells
*   **Title**: WELL COMPLETION 101 PART 2: WELL PERFORATION
    *   **Snippet**: Well perforation involves perforating casing, creating 4-8 holes per foot, and using charges to damage the formation, with 60, 90, and 120 degree spreads.
    *   **URL**: https://www.enverus.com/blog/well-completion-101-part-2-well-perforation/
*   **Title**: Investigating the effect of wellbore perforation on sand ...
    *   **Snippet**: The results obtained from the analysis showed that at a certain depth of the well, the more perforations there are, the higher the sand production rate.
    *   **URL**: https://www.nature.com/articles/s41598-025-04411-8


## First-Pass Reading - Key Findings

### Reservoir Drive Mechanisms [[ref]](https://www.slb.com/resource-library/oilfield-review/defining-series/defining-reservoir-drive-mechanisms)
*   **Drive Mechanisms**: Key types include depletion drive, waterdrive, gas cap drive, solution gas drive, and gravity drainage.
*   **Waterdrive Mechanism**: Characterized by energy from a connected aquifer. As hydrocarbons are extracted, water from the aquifer migrates into the reservoir, replacing the produced fluids. This mechanism is crucial for maintaining reservoir pressure and supporting production over longer periods.
    *   **Types**: Edge waterdrive and bottom waterdrive.
    *   **Efficiency Factors**: Dependent on aquifer size, rock properties, and fluid mobilities.
    *   **Key Data for Analysis**: Historically, reservoir pressure data and fluid production ratios (such as gas-oil ratio (GOR) and water-oil ratio (WOR)) are used to analyze drive mechanisms.

### Practical Guide for Feature Engineering of Time Series Data [[ref]](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/)
*   **Time-Based Features**: Derivable features such as year, month, day of month, day of week, day of year, hour, minute, and second. These features can capture cyclical and seasonal patterns.
*   **Lag Features**: Values of a variable from previous time steps (e.g., `value_t-1`, `value_t-7`). These are critical for capturing temporal dependencies in sequential data.
*   **Rolling Window Statistics**: Calculations over a defined moving window of past observations. Examples include rolling mean, median, standard deviation, minimum, and maximum. These help to smooth data, identify trends, and capture volatility over short-to-medium time frames (e.g., `rolling_mean_7`, `rolling_std_30`).
*   **Expanding Window Statistics**: Similar to rolling window statistics but the window expands to include all historical data up to the current point, capturing cumulative effects or long-term trends.
*   **Domain-Specific Features**: (Implicitly important) The article stresses the need for domain knowledge to create relevant features, although specific oil and gas examples aren't provided in the scraped content.

## Second-Pass Reading - Key Findings (Spatio-Temporal & Water Drive)

### Deep insight: an efficient hybrid model for oil well production forecasting using spatio-temporal convolutional networks and Kolmogorov–Arnold networks [[ref]](https://www.nature.com/articles/s41598-025-91412-2)
*   **Model**: TCN-KAN (Temporal Convolutional Networks and Kolmogorov–Arnold Networks).
*   **Approach**: Addresses spatio-temporal characteristics in oil well production forecasting. TCNs are used to capture temporal dependencies, and KANs are used for complex non-linear mappings and potentially interpreting underlying physical mechanisms.
*   **Potential Features**: The abstract mentions factors influencing fluid properties, which could be sources for feature engineering (e.g., downhole temperature, viscosity, gas-oil ratio).

### Forecasting multiple-well flow rates using a novel space-time modeling approach [[ref]](https://www.sciencedirect.com/science/article/abs/pii/S0920410520301212)
*   **Model**: Spatio-Temporal Autoregressive Moving Average (STARMA).
*   **Key Idea**: A new spatiotemporal approach for predicting production flow rates for a group of wells in a neighborhood. Emphasizes spatial clustering of wells to enhance forecasting.
*   **Relevance**: Suggests the importance of grouping wells and considering inter-well relationships for spatial features.

### Forecasting of oil production driven by reservoir spatial–temporal data based on normalized mutual information and Seq2Seq-LSTM [[ref]](https://journals.sagepub.com/doi/full/10.1177/01445987231188161)
*   **Problem Analysis**: Traditional ML struggles with changes in development measures (liquid production, well spacing density). Large historical data enables ML application. Need to account for production trends over time and data correlations (spatial/temporal).
*   **Model**: Seq2Seq-LSTM (Encoder-Decoder) for multivariate time series, capable of 'cross-series learning' to learn from multiple related wells/reservoirs simultaneously.
*   **Feature Engineering**: 
    *   **Feature Selection**: Normalized Mutual Information (NMI) is used to quantify nonlinear correlation between features and oil production. Highly correlated features identified were: liquid production, production time, equivalent well spacing density, fluidity, and original formation pressure.
    *   **Static Features**: Original formation pressure, reservoir thickness, reservoir area, porosity, heterogeneity, fluidity.
    *   **Dynamic Features**: Production time, liquid production, well spacing density.
    *   **Equivalent Well Spacing Density (Spatial Feature)**: Critical concept for standardizing different well types (horizontal/vertical). Horizontal wells are converted to an equivalent number of vertical wells based on total production and control area. Formula provided involves drainage radius, wellbore radius, skin factor, horizontal section length, effective thickness, horizontal and vertical permeability, distance from horizontal section to reservoir bottom. Then, equivalent well spacing density (wells/km^2) is calculated as (total equivalent vertical wells / reservoir area).
*   **Data Pre-processing**: Z-score standardization is applied to handle parameter unit differences.
*   **Modeling Water Drive Mechanism**: Explicitly states reservoirs are 

developed by edge water or bottom water drive because of the strong water energy.

The model captures how liquid production and well spacing density changes influence oil production, demonstrating that increased liquid production or increased equivalent well spacing density can improve oil production in such reservoirs. This directly models the 'water drive' impact by observing the system's response to these development measures.

### Oil well production prediction based on CNN-LSTM model with self-attention mechanism [[ref]](https://www.sciencedirect.com/science/article/abs/pii/S0360544223020959)
*   **Model**: CNN-LSTM-SA (Convolutional Neural Network - Long Short-Term Memory - Self-Attention).
*   **Key Idea**: CNN extracts spatio-temporal features, LSTM captures correlation information, and self-attention mechanism is used for enhanced feature weighting.
*   **Relevance**: Another strong deep learning architecture for spatio-temporal data, useful for integrating different types of features efficiently.