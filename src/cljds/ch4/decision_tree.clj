(ns cljds.ch4.decision-tree
  (:require [incanter.core :as i]
            [cljds.ch4.stats :refer :all]))

(defn map-vals [f coll]
  (into {} (map (fn [[k v]] [k (f v)]) coll)))

(defn gain-for-predictor [class-attr xs predictor]
  (let [grouped-classes (->> (group-by predictor xs)
                             (vals)
                             (map (partial map class-attr)))]
    (information-gain grouped-classes)))

(defn best-predictor [class-attr predictors xs]
  (let [gain (partial gain-for-predictor class-attr xs)]
    (when (seq predictors)
      (apply max-key gain predictors))))

(defn modal-class [classes]
  (->> (frequencies classes)
       (apply max-key val)
       (key)))

(defn decision-tree [class-attr predictors xs]
  (let [classes (map class-attr xs)]
    (if (zero? (entropy classes))
      (first classes)
      (if-let [predictor (best-predictor class-attr
                                         predictors xs)]
        (let [predictors  (remove #{predictor} predictors)
              tree-branch (partial decision-tree
                                   class-attr predictors)]
          (->> (group-by predictor xs)
               (map-vals tree-branch)
               (vector predictor)))
        (modal-class classes)))))

(defn tree-classify [model test]
  (if (vector? model)
    (let [[predictor branches] model
          branch (get branches (get test predictor))]
      (recur branch test))
    model))
