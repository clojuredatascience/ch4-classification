(ns cljds.ch4.stats
  (:require [clojure.set :as set]
            [incanter.core :as i]
            [incanter.optimize :as o]
            [incanter.stats :as s]))

(defn relative-risk [p1 p2]
  (float (/ p1 p2)))

(defn odds-ratio [p1 p2]
  (float
   (/ (* p1 (- 1 p2))
      (* p2 (- 1 p1)))))

(defn se-proportion [p n]
  (-> (- 1 p)
      (* p)
      (/ n)
      (i/sqrt)))

(defn se-large-proportion [p n N]
  (* (se-proportion p n)
     (i/sqrt (/ (- N n)
                (- n 1)))))

(defn chisq-stat [observed expected]
  (let [f (fn [observed expected]
            (/ (i/sq (- observed expected)) expected))]
    (reduce + (map f observed expected))))

(defn confusion-matrix [ys y-hats]
  (let [classes   (into #{} (concat ys y-hats))
        confusion (frequencies (map vector ys y-hats))]
    (i/dataset (cons nil classes)
               (for [x classes]
                 (cons x
                       (for [y classes]
                         (get confusion [x y])))))))

(defn kappa-statistic [ys y-hats]
  (let [n (count ys)
        test-class (first ys)
        pa  (/ (count (filter true? (map = ys y-hats))) n)
        ey  (/ (count (filter #(= test-class %) ys)) n)
        eyh (/ (count (filter #(= test-class %) y-hats)) n)
        pe (+ (* ey eyh)
              (* (- 1 ey)
                 (- 1 eyh)))]
    (float (/ (- pa pe)
              (- 1 pe)))))

(defn information [x]
  (- (i/log2 x)))

(defn entropy [xs]
  (let [n (count xs)
        f (fn [x]
            (let [p (/ x n)]
              (* p (information p))))]
    (->> (frequencies xs)
         (vals)
         (map f)
         (reduce +))))

(defn weighted-entropy [groups]
  (let [n (count (apply concat groups))
        e (fn [group]
            (* (entropy group)
               (/ (count group) n)))]
    (->> (map e groups)
         (reduce +))))

(defn information-gain [groups]
  (- (entropy (apply concat groups))
     (weighted-entropy groups)))
