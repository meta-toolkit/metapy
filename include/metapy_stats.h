/**
 * @file metapy_stats.h
 * @author Chase Geigle
 */

#ifndef METAPY_STATS_H_
#define METAPY_STATS_H_

#include <cmath>
#include <pybind11/pybind11.h>

#include "meta/stats/multinomial.h"

/**
 * Wrapper class for stats::multinomial<T> for Python. This makes it so we
 * don't have to bind stats::multinomial multiple times for each T we want
 * to use. Instead, we just need to convert it to a py_multinomial at the
 * return site in the python binding function.
 */
class py_multinomial
{
  public:
    template <class T>
    py_multinomial(const meta::stats::multinomial<T>& dist)
        : concept_{meta::make_unique<multinomial_impl<T>>(dist)}
    {
        // nothing
    }

    void increment(pybind11::object obj, double count)
    {
        concept_->increment(obj, count);
    }

    void decrement(pybind11::object obj, double count)
    {
        concept_->decrement(obj, count);
    }

    double counts(pybind11::object obj) const
    {
        return concept_->counts(obj);
    }

    double counts() const
    {
        return concept_->counts();
    }

    uint64_t unique_events() const
    {
        return concept_->unique_events();
    }

    void each_seen_event(std::function<void(const pybind11::object&)> fun) const
    {
        concept_->each_seen_event(fun);
    }

    void clear()
    {
        concept_->clear();
    }

    double probability(pybind11::object obj) const
    {
        return concept_->probability(obj);
    }

  private:
    class multinomial_concept
    {
      public:
        virtual ~multinomial_concept() = default;
        virtual void increment(pybind11::object obj, double count) = 0;
        virtual void decrement(pybind11::object obj, double count) = 0;
        virtual double counts(pybind11::object obj) const = 0;
        virtual double counts() const = 0;
        virtual uint64_t unique_events() const = 0;
        virtual void each_seen_event(
            std::function<void(const pybind11::object&)> fun) const = 0;
        virtual void clear() = 0;
        virtual double probability(pybind11::object obj) const = 0;
    };

    template <class T>
    class multinomial_impl : public multinomial_concept
    {
      public:
        multinomial_impl(const meta::stats::multinomial<T>& dist) : dist_{dist}
        {
            // nothing
        }

        void increment(pybind11::object obj, double count) override
        {
            dist_.increment(obj.cast<T>(), count);
        }

        void decrement(pybind11::object obj, double count) override
        {
            dist_.decrement(obj.cast<T>(), count);
        }

        double counts(pybind11::object obj) const override
        {
            return dist_.counts(obj.cast<T>());
        }

        double counts() const override
        {
            return dist_.counts();
        }

        uint64_t unique_events() const override
        {
            return dist_.unique_events();
        }

        void each_seen_event(
            std::function<void(const pybind11::object&)> fun) const override
        {
            dist_.each_seen_event(
                [&](const T& event) { fun(pybind11::cast(event)); });
        }

        void clear() override
        {
            dist_.clear();
        }

        double probability(pybind11::object obj) const override
        {
            return dist_.probability(obj.cast<T>());
        }

      private:
        meta::stats::multinomial<T> dist_;
    };

    std::unique_ptr<multinomial_concept> concept_;
};

void metapy_bind_stats(pybind11::module& m);
#endif
