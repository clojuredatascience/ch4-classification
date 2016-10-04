(ns cljds.ch4.logistic
  (:require [incanter.core :as i]
            [incanter.optimize :as o]
            [incanter.stats :as s]))

(defn logistic-class [x]
  (if (>= x 0.5) 1 0))

(defn sigmoid-function [coefs]
  (let [bt (i/trans coefs)
        z  (fn [x] (- (first (i/mmult bt x))))]
    (fn [x]
      (/ 1
         (+ 1
            (i/exp (z x)))))))

(defn logistic-cost [ys y-hats]
  (let [cost (fn [y y-hat]
               (if (zero? y)
                 (- (i/log (- 1 y-hat)))
                 (- (i/log y-hat))))]
    (s/mean (map cost ys y-hats))))

(defn logistic-regression [ys xs]
  (let [cost-fn (fn [coefs]
                  (let [classify (sigmoid-function coefs)
                        y-hats   (map (comp classify i/trans) xs)]
                    (logistic-cost ys y-hats)))
        init-coefs (repeat (i/ncol xs) 0.0)]
    (-> (o/minimize cost-fn init-coefs)
        (:value)
        (i/to-vect))))
