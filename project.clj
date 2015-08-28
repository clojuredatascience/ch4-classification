(defproject cljds/ch4 "0.1.0"
  :description "Example code for the book Clojure for Data Science"
  :url "https://github.com/clojuredatascience/ch4-classification"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [incanter/incanter "1.5.5"]
                 [cc.artifice/clj-ml "0.5.1"]]
  :resource-paths ["resources" "data"]
  :aot [cljds.ch4.core]
  :main cljds.ch4.core
  :repl-options {:init-ns cljds.ch4.examples}
  :profiles {:dev {:dependencies [[org.clojure/tools.cli "0.3.1"]]}})
