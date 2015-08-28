(ns cljds.ch4.examples
  (:require [cljds.ch4.bayes :refer :all]
            [cljds.ch4.data :refer :all]
            [cljds.ch4.decision-tree :refer :all]
            [cljds.ch4.logistic :refer :all]
            [cljds.ch4.stats :refer :all]
            [clj-ml.classifiers :as cl]
            [clj-ml.data :as mld]
            [clj-ml.filters :as mlf]
            [clj-ml.utils :as clu]
            [clojure.java.io :as io]
            [incanter.charts :as c]
            [incanter.core :as i]
            [incanter.optimize :as o]
            [incanter.stats :as s]
            [incanter.svg :as svg]))

(defn ex-4-1 []
  (i/view (load-data "titanic.tsv")))

(defn ex-4-2 []
  (->> (load-data "titanic.tsv")
       (frequency-table :count [:sex :survived])))

(defn ex-4-3 []
  (->> (load-data "titanic.tsv")
       (frequency-map :count [:sex :survived])))

(defn ex-4-4 []
  (-> (load-data "titanic.tsv")
      (fatalities-by-sex)))

(defn ex-4-5 []
  (let [proportions (-> (load-data "titanic.tsv")
                        (fatalities-by-sex))]
    (relative-risk (get proportions :male)
                   (get proportions :female))))

(defn ex-4-6 []
  (let [proportions (-> (load-data "titanic.tsv")
                        (fatalities-by-sex))]
    (odds-ratio (get proportions :male)
                (get proportions :female))))

(defn ex-4-7 []
  (let [passengers (concat (repeat 127 0)
                           (repeat 339 1))
        bootstrap (s/bootstrap passengers i/sum :size 10000)]
    (-> (c/histogram bootstrap
                     :x-label "Female Survivors"
                     :nbins 20)
        (i/view))))

(defn ex-4-8 []
  (-> (concat (repeat 127 0)
              (repeat 339 1))
      (s/bootstrap i/sum :size 10000)
      (s/sd)))

(defn ex-4-9 []
  (let [passengers (concat (repeat 127 0)
                           (repeat 339 1))
        bootstrap (s/bootstrap passengers i/sum :size 10000)
        binomial (fn [x]
                   (s/pdf-binomial x :size 466 :prob (/ 339 466)))
        normal (fn [x]
                 (s/pdf-normal x :mean 339 :sd 9.57))]
    (-> (c/histogram bootstrap
                     :x-label "Female Survivors"
                     :series-label "Bootstrap"
                     :nbins 20
                     :density true
                     :legend true)
        (c/add-function binomial 300 380
                        :series-label "Biomial")
        (c/add-function normal 300 380
                        :series-label "Normal")
        (i/view))))

(defn ex-4-10 []
  (let [survived (->> (load-data "titanic.tsv")
                      (frequency-map :count [:sex :survived]))
        n (reduce + (vals (get survived "female")))
        p (/ (get-in survived ["female" "y"]) n)]
    (se-proportion p n)))

(defn ex-4-11 []
  (let [dataset     (load-data "titanic.tsv")
        proportions (fatalities-by-sex dataset)
        survived    (frequency-map :count [:survived] dataset)
        total  (reduce + (vals survived))
        pooled (/ (get survived "n") total)
        p-diff (- (get proportions :male)
                  (get proportions :female))
        z-stat (/ p-diff (se-proportion pooled total))]
    (- 1 (s/cdf-normal (i/abs z-stat)))))


(defn ex-4-12 []
  (->> (load-data "titanic.tsv")
       (frequency-table :count [:survived :pclass])))

(defn ex-4-13 []
  (let [data (->> (load-data "titanic.tsv")
                  (frequency-table :count [:survived :pclass]))]
    (-> (c/stacked-bar-chart :pclass :count
                             :group-by :survived
                             :legend true
                             :x-label "Class"
                             :y-label "Passengers"
                             :title "Survival of Titanic Passengers by Class"
                             :data data)
        (i/view))))

(defn ex-4-14 []
  (-> (load-data "titanic.tsv")
      (expected-frequencies)))

(defn ex-4-15 []
  (-> (load-data "titanic.tsv")
      (observed-frequencies)))

(defn ex-4-16 []
  (let [data (load-data "titanic.tsv")
        observed (observed-frequencies data)
        expected (expected-frequencies data)]
    (float (chisq-stat observed expected))))

(defn ex-4-17 []
  (let [data (load-data "titanic.tsv")
        observed (observed-frequencies data)
        expected (expected-frequencies data)
        x2-stat  (chisq-stat observed expected)]
    (s/cdf-chisq x2-stat :df 2 :lower-tail? false)))

(defn ex-4-18 []
  (let [table  (->> (load-data "titanic.tsv")
                    (frequency-table :count [:pclass :survived])
                    (i/$order [:survived :pclass] :asc))
        frequencies (i/$ :count table)
        matrix (i/matrix frequencies 3)]
    (println "Observed:"     table)
    (println "Frequencies:"  frequencies)
    (println "Observations:" matrix)
    (println "Chi-Squared test:")
    (-> (s/chisq-test :table matrix)
        (clojure.pprint/pprint))))

;; Logistic Regression

(defn ex-4-19 []
  (let [f (fn [[x]]
            (i/sq x))
        init [10]]
    (o/minimize f init)))

(defn ex-4-20 []
   (let [f (fn [[x]]
             (i/sin x))]
     (println (:value (o/minimize f [1])))
     (println (:value (o/minimize f [10])))
     (println (:value (o/minimize f [100])))))

(defn ex-4-21 []
   (let [data (matrix-dataset)
         ys (i/$ 0 data)
         xs (i/$ [:not 0] data)]
     (logistic-regression ys xs)))


(defn ex-4-22 []
  (let [data (matrix-dataset)
        ys (i/$ 0 data)
        xs (i/$ [:not 0] data)
        coefs (logistic-regression ys xs)
        classifier (comp logistic-class
                      (sigmoid-function coefs)
                      i/trans)]
    (println "Observed: " (map int (take 10 ys)))
    (println "Predicted:" (map classifier (take 10 xs)))))

(defn ex-4-23 []
   (let [data (matrix-dataset)
         ys (i/$ 0 data)
         xs (i/$ [:not 0] data)
         coefs (logistic-regression ys xs)
         classifier (comp logistic-class
                       (sigmoid-function coefs)
                       i/trans)
         y-hats (map classifier xs)]
     (frequencies (map = y-hats (map int ys)))))

(defn ex-4-24 []
   (let [data (matrix-dataset)
         ys (i/$ 0 data)
         xs (i/$ [:not 0] data)
         coefs (logistic-regression ys xs)
         classifier (comp logistic-class
                       (sigmoid-function coefs)
                       i/trans)
         y-hats (map classifier xs)]
     (confusion-matrix (map int ys) y-hats)))

(defn ex-4-25 []
   (let [data (matrix-dataset)
         ys (i/$ 0 data)
         xs (i/$ [:not 0] data)
         coefs (logistic-regression ys xs)
         classifier (comp logistic-class
                       (sigmoid-function coefs)
                       i/trans)
         y-hats (map classifier xs)]
     (float (kappa-statistic (map int ys) y-hats))))

(defn ex-4-26 []
   (->> (load-data "titanic.tsv")
        (:rows)
        (bayes-classifier :survived [:sex :pclass])
        (clojure.pprint/pprint)))

(defn ex-4-27 []
   (let [model (->> (load-data "titanic.tsv")
                    (:rows)
                    (bayes-classifier :survived [:sex :pclass]))]
     (println "Third class male:"
              (bayes-classify model {:sex "male" :pclass "third"}))
     (println "First class female:"
              (bayes-classify model {:sex "female" :pclass "first"}))))

(defn ex-4-28 []
   (let [data (:rows (load-data "titanic.tsv"))
         model (bayes-classifier :survived [:sex :pclass] data)
         test (fn [test]
                (= (:survived test)
                   (bayes-classify model
                            (select-keys test [:sex :class]))))
         results (frequencies (map test data))]
     (/ (get results true)
        (apply + (vals results)))))


(defn ex-4-29 []
   (let [data (:rows (load-data "titanic.tsv"))
         model (bayes-classifier :survived [:sex :pclass] data)
         classify (fn [test]
                    (->> (select-keys test [:sex :pclass])
                         (bayes-classify model)))
         ys      (map :survived data)
         y-hats (map classify data)]
     (confusion-matrix ys y-hats)))

(defn ex-4-30 []
   (let [red-black (concat (repeat 26 1)
                           (repeat 26 0))]
     (entropy red-black)))

(defn ex-4-31 []
   (let [picture-not-picture (concat (repeat 12 1)
                                     (repeat 40 0))]
     (entropy picture-not-picture)))


(defn ex-4-32 []
   (->> (load-data "titanic.tsv")
        (:rows)
        (map :survived)
        (entropy)))

(defn ex-4-33 []
   (->> (load-data "titanic.tsv")
        (:rows)
        (group-by :sex)
        (vals)
        (map (partial map :survived))
        (weighted-entropy)))

(defn ex-4-34 []
   (->> (load-data "titanic.tsv")
        (:rows)
        (group-by :pclass)
        (vals)
        (map (partial map :survived))
        (information-gain)))

(defn ex-4-35 []
   (->> (load-data "titanic.tsv")
        (:rows)
        (best-predictor :survived [:sex :pclass])))

(defn ex-4-36 []
   (->> (load-data "titanic.tsv")
        (:rows)
        (decision-tree :survived [:pclass :sex])
        (clojure.pprint/pprint)))

(defn ex-4-37 []
   (let [data (load-data "titanic.tsv")]
     (->> (i/transform-col data :age age-categories)
          (:rows)
          (decision-tree :survived [:pclass :sex :age])
          (clojure.pprint/pprint))))

(defn ex-4-38 []
   (let [data (-> (load-data "titanic.tsv")
                  (i/transform-col :age age-categories)
                  (:rows))
         tree (decision-tree :survived [:pclass :sex :age] data)
         test {:sex "male" :pclass "second" :age "child"}]
     (tree-classify tree test)))

(defn ex-4-39 []
   (let [data (-> (load-data "titanic.tsv")
                  (i/transform-col :age age-categories)
                  (:rows))
         tree (decision-tree :survived [:pclass :sex :age] data)]
     (confusion-matrix (map :survived data)
                       (map (partial tree-classify tree) data))))

(defn ex-4-40 []
   (let [data (-> (load-data "titanic.tsv")
                  (i/transform-col :age age-categories)
                  (:rows))
         tree (decision-tree :survived [:pclass :sex :age] data)
         ys     (map :survived data)
         y-hats (map (partial tree-classify tree) data)]
     (float (kappa-statistic ys y-hats))))

(defn ex-4-41 []
  (let [data (-> (load-data "titanic.tsv")
                 (:rows))
        tree (decision-tree :survived
                            [:pclass :sex :age :fare] data)
        ys     (map :survived data)
        y-hats (map (partial tree-classify tree) data)]
    (float (kappa-statistic ys y-hats))))

(defn ex-4-42 []
  (let [dataset (to-weka (load-data "titanic.tsv"))
        classifier (-> (cl/make-classifier :decision-tree :c45)
                       (cl/classifier-train dataset))
        classify (partial cl/classifier-classify classifier)
        ys     (map str  (mld/dataset-class-values dataset))
        y-hats (map name (map classify dataset))]
    (println "Confusion:" (confusion-matrix ys y-hats))
    (println "Kappa:" (kappa-statistic ys y-hats))))


(defn ex-4-43 []
  (let [[test-set train-set] (-> (load-data "titanic.tsv")
                                 (to-weka)
                                 (mld/do-split-dataset :percentage
                                                       30))
        classifier (-> (cl/make-classifier :decision-tree :c45)
                       (cl/classifier-train train-set))
        classify (partial cl/classifier-classify classifier)
        ys     (map str  (mld/dataset-class-values test-set))
        y-hats (map name (map classify test-set))]
    (println "Confusion:" (confusion-matrix ys y-hats))
    (println "Kappa:" (kappa-statistic ys y-hats))))

(defn ex-4-44 []
  (let [[train-set test-set] (-> (load-data "titanic.tsv")
                                 (to-weka)
                                 (mld/do-split-dataset :percentage
                                                       70))
        classifier (-> (cl/make-classifier :decision-tree :c45)
                       (cl/classifier-train train-set))
        classify (partial cl/classifier-classify classifier)
        ys     (map str  (mld/dataset-class-values test-set))
        y-hats (map name (map classify test-set))]
    (println "Kappa:" (kappa-statistic ys y-hats))))

(defn ex-4-45 []
  (let [dataset (-> (load-data "titanic.tsv")
                    (to-weka))
         classifier (-> (cl/make-classifier :decision-tree :c45)
                        (cl/classifier-train dataset))
         evaluation (cl/classifier-evaluate classifier
                                            :cross-validation
                                            dataset 10)]
     (println (:confusion-matrix evaluation))
     (println (:summary evaluation))))

(defn ex-4-46 []
  (let [dataset (->> (load-data "titanic.tsv")
                     (to-weka)
                     (mlf/make-apply-filter
                      :replace-missing-values {}))
        classifier (-> (cl/make-classifier :decision-tree :c45)
                       (cl/classifier-train dataset))
        evaluation (cl/classifier-evaluate classifier
                                           :cross-validation
                                           dataset 10)]
    (println (:kappa evaluation))))


(defn ex-4-47 []
  (let [dataset (->> (load-data "titanic.tsv")
                     (to-weka)
                     (mlf/make-apply-filter
                      :replace-missing-values {}))
        classifier (cl/make-classifier :decision-tree
                                       :random-forest)
        evaluation (cl/classifier-evaluate classifier
                                           :cross-validation
                                           dataset 10)]
    (println (:confusion-matrix evaluation))
    (println (:summary evaluation))))

(defn ex-4-48 []
  (let [dataset (->> (load-data "titanic.tsv")
                     (to-weka)
                     (mlf/make-apply-filter
                      :replace-missing-values {}))
        classifier (cl/make-classifier :decision-tree
                                       :random-forest)
        file (io/file (io/resource "classifier.bin"))]
    (clu/serialize-to-file classifier file)))

