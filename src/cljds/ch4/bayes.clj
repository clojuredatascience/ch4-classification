(ns cljds.ch4.bayes)

(defn inc-class-total [model class]
  (update-in model [:classes class :n] (fnil inc 0)))

(defn inc-predictors-count-fn [row class]
  (fn [model attr]
    (let [val (get row attr)]
      (update-in model [:classes class :predictors attr val] (fnil inc 0)))))

(defn assoc-row-fn [class-attr predictors]
  (fn [model row]
    (let [class (get row class-attr)]
      (reduce (inc-predictors-count-fn row class)
              (inc-class-total model class)
              predictors))))

(defn bayes-classifier [class-attr predictors dataset]
  (reduce (assoc-row-fn class-attr predictors) {:n (count dataset)} dataset))

(defn posterior-probability [model test class-attr]
  (let [observed (get-in model [:classes class-attr])
        prior (/ (:n observed)
                 (:n model))]
    (apply * prior
           (for [[predictor value] test]
             (/ (get-in observed [:predictors predictor value])
                (:n observed))))))

(defn bayes-classify [model test]
  (let [probability (partial posterior-probability model test)
        classes     (keys (:classes model))]
    (apply max-key probability classes)))
