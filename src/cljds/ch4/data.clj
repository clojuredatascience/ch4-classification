(ns cljds.ch4.data
  (:require [incanter.core :as i]
            [incanter.io :as iio] 
            [clojure.java.io :as io]
            [clj-ml.io :as cio]
            [incanter.excel :as xls]
            [clj-ml.data :as mld]
            [clj-ml.filters :as mlf]))

(defn load-data [file]
  (-> (io/resource file)
      (str)
      (iio/read-dataset :delim \tab :header true)))

(defn frequency-table [sum-column group-columns dataset]
  (->> (i/$ group-columns dataset)
       (i/add-column sum-column (repeat 1))
       (i/$rollup :sum sum-column group-columns)))

(defn frequency-map [sum-column group-cols dataset]
  (let [f (fn [freq-map row]
            (let [groups (map row group-cols)]
              (->> (get row sum-column)
                   (assoc-in freq-map groups))))]
    (->> (frequency-table sum-column group-cols dataset)
         (:rows)
         (reduce f {}))))

(defn fatalities-by-sex [dataset]
  (let [totals (frequency-map :count [:sex] dataset)
        groups (frequency-map :count [:sex :survived] dataset)]
    {:male (/ (get-in groups ["male" "n"])
              (get totals "male"))
     :female (/ (get-in groups ["female" "n"])
                (get totals "female"))}))

(defn expected-frequencies [data]
  (let [as (vals (frequency-map :count [:survived] data))
        bs (vals (frequency-map :count [:pclass] data))
        total (-> data :rows count)]
    (for [a as
          b bs]
      (* a (/ b total)))))

(defn observed-frequencies [data]
  (let [as (frequency-map :count [:survived] data)
        bs (frequency-map :count [:pclass] data)
        actual (frequency-map :count [:survived :pclass] data)]
    (for [a (keys as)
          b (keys bs)]
      (get-in actual [a b]))))

(defn add-dummy [column-name from-column value dataset]
  (i/add-derived-column column-name
                        [from-column]
                        #(if (= % value) 1 0)
                        dataset))

(defn matrix-dataset []
  (->> (load-data "titanic.tsv")
       (add-dummy :dummy-survived :survived "y")
       (i/add-column :bias (repeat 1.0))
       (add-dummy :dummy-mf :sex "male")
       (add-dummy :dummy-1 :pclass "first")
       (add-dummy :dummy-2 :pclass "second")
       (add-dummy :dummy-3 :pclass "third")
       (i/$ [:dummy-survived :bias :dummy-mf
             :dummy-1 :dummy-2 :dummy-3])
       (i/to-matrix)))

(defn age-categories [age]
   (cond
     (nil? age) "unknown"
     (< age 13) "child"
     :default   "adult"))

(defn with-survived-dummy [data]
  (add-dummy :survived-d :survived "y" data))

(defn distinct-values [column-name dataset]
  (->> (i/$ column-name dataset)
       (distinct)))

(defn plug-age [age]
  (or age 30))

(defn plug-fare [fare]
  (or fare 13.3))

(defn multinomial [power]
  (fn [x]
    (i/pow x power)))

(defn to-weka [dataset]
  (let [attributes [{:survived ["y" "n"]}
                    {:pclass ["first" "second" "third"]}
                    {:sex ["male" "female"]}
                    :age
                    :fare]
        vectors (->> dataset
                     (i/$ [:survived :pclass :sex :age :fare])
                     (i/to-vect))]
    (mld/make-dataset :titanic-weka attributes vectors
                      {:class :survived})))


(defn load-weka [file]
  (-> (cio/load-instances :csv (str (io/resource file)))
      (mld/dataset-set-class :survived)
      (mlf/numeric-to-nominal {:attributes [:survived]})))
